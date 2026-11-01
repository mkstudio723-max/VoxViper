// BLACK TUNE RECORD v3
// - YIN pitch tracking with median filter, hysteresis and hold (precise, no flicker)
// - PSOLA engine: output pitch set directly from the target note (exact), formants kept
// - Premium custom GUI: LED tuner, glossy keyboard, metal knobs, presets
#include <juce_audio_processors/juce_audio_processors.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <utility>
#include <vector>

static juce::StringArray keyNames()
{
    return { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
}
static juce::StringArray scaleNames()
{
    return { "Chromatic", "Major", "Minor", "Minor Pentatonic", "Harmonic Minor" };
}
// bit k set => semitone k above the key is allowed
static int scaleMask(int idx)
{
    static const int masks[5] = { 0xFFF, 0xAB5, 0x5AD, 0x4A9, 0x9AD };
    return masks[juce::jlimit(0, 4, idx)];
}
static float softClip(float x)
{
    const float a = std::abs(x);
    if (a < 0.9f) return x;
    const float s = 0.9f + 0.1f * std::tanh((a - 0.9f) / 0.1f);
    return x < 0.f ? -s : s;
}

//==============================================================================
class BlackTune : public juce::AudioProcessor
{
public:
    BlackTune()
        : AudioProcessor(BusesProperties()
                             .withInput("Input", juce::AudioChannelSet::stereo(), true)
                             .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "PARAMS", createLayout())
    {
    }

    const juce::String getName() const override { return "BLACK TUNE RECORD"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    bool isBusesLayoutSupported(const BusesLayout& l) const override
    {
        auto out = l.getMainOutputChannelSet();
        return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo())
               && out == l.getMainInputChannelSet();
    }

    void getStateInformation(juce::MemoryBlock& dest) override
    {
        if (auto xml = apvts.copyState().createXml())
            copyXmlToBinary(*xml, dest);
    }
    void setStateInformation(const void* data, int size) override
    {
        if (auto xml = getXmlFromBinary(data, size))
            apvts.replaceState(juce::ValueTree::fromXml(*xml));
    }

    void prepareToPlay(double sampleRate, int) override
    {
        sr = sampleRate;
        delay0 = (int) std::ceil(2.95 * sr / 90.0);   // fixed latency of the PSOLA engine
        setLatencySamples(delay0);

        for (auto& v : dl)   v.assign(kBuf, 0.f);
        for (auto& v : outB) v.assign(kBuf, 0.f);
        for (auto& v : comb) v.assign(kBuf, 0.f);
        det.assign(kBuf, 0.f);
        lp.assign(kBuf, 0.f);
        scratch.assign((size_t) kBuf, 0.f);
        yin.assign((size_t) kBuf, 0.f);

        wp = 0; hopCount = 0;
        hop = juce::jmax(64, (int) (0.0058 * sr));
        T0cur = (float) (sr / 200.0);
        curMidiOut = targetMidiOut = 69.f + 12.f * std::log2(200.f / 440.f);
        outFreq = 200.f;
        nextTs = 0.0;
        anaMark = nextTs - (double) delay0;
        combT = (float) (sr / 200.0);
        lpState = 0.f;
        snapOn = false;
        histN = 0; histIdx = 0; unvoicedHops = 0; curNote = -1;
        meterVoiced = false;
    }
    void releaseResources() override {}

    void processBlock(juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int nCh = juce::jmin(2, buf.getNumChannels());
        const int n = buf.getNumSamples();

        const float amount = *apvts.getRawParameterValue("amount");
        const float retMs  = *apvts.getRawParameterValue("retune");
        const float robot  = *apvts.getRawParameterValue("robot");
        const float grit   = *apvts.getRawParameterValue("grit");
        const float mix    = *apvts.getRawParameterValue("mix");
        const float outDb  = *apvts.getRawParameterValue("output");
        keyIdx   = (int) *apvts.getRawParameterValue("key");
        scaleIdx = (int) *apvts.getRawParameterValue("scale");
        amountP = amount;

        const float minCoef = 1.f - std::exp(-1.f / (0.0015f * (float) sr));
        const float coef = retMs <= 0.01f ? minCoef
                         : juce::jmax(minCoef, 1.f - std::exp(-1.f / (retMs * 0.001f * (float) sr)));
        const float releaseCoef = 1.f - std::exp(-1.f / (0.02f * (float) sr));
        const float lpA = 1.f - std::exp(-juce::MathConstants<float>::twoPi * 900.f / (float) sr);

        const float drive = 1.f + grit * 6.f;
        const float invDrive = 1.f / std::sqrt(drive);
        const float levels = std::pow(2.f, 16.f - grit * 8.f);
        const float fb = robot * 0.7f;
        const float combNorm = 1.f / (1.f + fb * 0.6f);
        const float outGain = juce::Decibels::decibelsToGain(outDb);

        float pk = 0.f;

        for (int i = 0; i < n; ++i)
        {
            float mono = 0.f;
            for (int c = 0; c < nCh; ++c)
            {
                const float in = buf.getReadPointer(c)[i];
                dl[(size_t) c][(size_t) (wp & kMask)] = in;
                mono += in;
            }
            mono /= (float) nCh;
            det[(size_t) (wp & kMask)] = mono;
            lpState += lpA * (mono - lpState);
            lp[(size_t) (wp & kMask)] = lpState;

            if (++hopCount >= hop) { hopCount = 0; detect(); }

            curMidiOut += (meterVoiced.load() ? coef : releaseCoef) * (targetMidiOut - curMidiOut);
            curMidiOut = juce::jlimit(20.f, 120.f, curMidiOut);

            generateGrains(nCh);

            combT += 0.002f * ((float) (sr / (double) outFreq) - combT);
            combT = juce::jlimit(8.f, (float) (kBuf - 4), combT);

            for (int c = 0; c < nCh; ++c)
            {
                float* io = buf.getWritePointer(c);

                float wet = outB[(size_t) c][(size_t) (wp & kMask)];
                outB[(size_t) c][(size_t) (wp & kMask)] = 0.f;

                const float yc = wet + fb * readAt(comb[(size_t) c], (double) wp - (double) combT);
                comb[(size_t) c][(size_t) (wp & kMask)] = yc;
                wet = yc * combNorm;

                if (grit > 0.001f)
                {
                    wet = std::tanh(wet * drive) * invDrive;
                    wet = std::round(wet * levels) / levels;
                }

                const float dry = dl[(size_t) c][(size_t) ((wp - delay0) & kMask)];
                float out = (dry * (1.f - mix) + wet * mix) * outGain;
                out = softClip(out);
                io[i] = out;
                pk = juce::jmax(pk, std::abs(out));
            }
            ++wp;
        }
        outLevel.store(juce::jmax(pk, outLevel.load() * 0.9f));
    }

    // shared with the editor
    juce::AudioProcessorValueTreeState apvts;
    std::atomic<float> meterDetected { 0.f };
    std::atomic<float> meterTarget { 0.f };
    std::atomic<bool> meterVoiced { false };
    std::atomic<float> outLevel { 0.f };

private:
    static constexpr int kBuf = 1 << 15;
    static constexpr int kMask = kBuf - 1;

    double sr = 44100.0;
    int delay0 = 1500, hopCount = 0, hop = 256;
    long long wp = 0;
    double nextTs = 0.0, anaMark = 0.0;
    float T0cur = 220.f, curMidiOut = 55.f, targetMidiOut = 55.f, outFreq = 200.f, combT = 200.f;
    float lpState = 0.f;
    bool snapOn = false;
    int keyIdx = 0, scaleIdx = 0;
    float amountP = 1.f;
    float hist[5] = { 0, 0, 0, 0, 0 };
    int histN = 0, histIdx = 0, unvoicedHops = 0, curNote = -1;
    std::array<std::vector<float>, 2> dl, outB, comb;
    std::vector<float> det, lp, scratch, yin;

    float readAt(const std::vector<float>& b, double pos) const
    {
        const long long i0 = (long long) std::floor(pos);
        const float f = (float) (pos - (double) i0);
        const float a = b[(size_t) (i0 & kMask)];
        const float c = b[(size_t) ((i0 + 1) & kMask)];
        return a + f * (c - a);
    }

    // next analysis mark one period after 'a', snapped to the nearest waveform peak when voiced
    double nextAnaMark(double a, double T) const
    {
        const double nominal = a + T;
        if (!snapOn) return nominal;
        const long long lo = (long long) std::floor(nominal - 0.25 * T);
        const long long hi = (long long) std::ceil(nominal + 0.25 * T);
        long long bestI = (long long) std::llround(nominal);
        float bestV = -1e9f;
        for (long long i = lo; i <= hi; ++i)
        {
            const float v = lp[(size_t) (i & kMask)];
            if (v > bestV) { bestV = v; bestI = i; }
        }
        return (double) bestI;
    }

    // PSOLA: grains of two periods, placed exactly at the spacing of the TARGET note
    void generateGrains(int nCh)
    {
        const double T = (double) T0cur;
        if (nextTs < (double) (wp - 4096)) nextTs = (double) wp;

        while (nextTs <= (double) wp + 1.15 * T)
        {
            const int Ti = juce::jmax(8, (int) std::llround(T));
            const double tdes = nextTs - (double) delay0;

            int guard = 0;
            while (std::abs(anaMark + T - tdes) < std::abs(anaMark - tdes) && guard++ < 16)
                anaMark = nextAnaMark(anaMark, T);
            if (guard >= 16) anaMark = tdes;

            if (anaMark + (double) Ti + 2.0 > (double) wp) anaMark = (double) (wp - Ti - 2);
            if (anaMark - (double) Ti < (double) (wp - kBuf + 8)) anaMark = (double) (wp - kBuf + 8 + Ti);

            const double curFreq = 440.0 * std::pow(2.0, ((double) curMidiOut - 69.0) / 12.0);
            const double Ts = juce::jlimit(sr / 2000.0, sr / 40.0, sr / curFreq);
            outFreq = (float) curFreq;
            const float scale = (float) (Ts / T);

            const long long tsInt = (long long) std::floor(nextTs + 0.5);
            const double frac = nextTs - (double) tsInt;

            for (int k = -Ti; k <= Ti; ++k)
            {
                const float w = (0.5f + 0.5f * std::cos(juce::MathConstants<float>::pi * (float) k / (float) Ti)) * scale;
                for (int c = 0; c < nCh; ++c)
                    outB[(size_t) c][(size_t) ((tsInt + k) & kMask)] += w * readAt(dl[(size_t) c], anaMark + (double) k - frac);
            }
            nextTs += Ts;
        }
    }

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout l;
        l.add(std::make_unique<AudioParameterChoice>(ParameterID{"key", 1}, "Key", keyNames(), 0));
        l.add(std::make_unique<AudioParameterChoice>(ParameterID{"scale", 1}, "Scale", scaleNames(), 2));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"retune", 1}, "Retune Speed",
              NormalisableRange<float>(0.f, 200.f, 0.1f, 0.5f), 0.f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"amount", 1}, "Amount",
              NormalisableRange<float>(0.f, 1.f, 0.001f), 1.f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"robot", 1}, "Robot",
              NormalisableRange<float>(0.f, 1.f, 0.001f), 0.3f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"grit", 1}, "Grit",
              NormalisableRange<float>(0.f, 1.f, 0.001f), 0.f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"mix", 1}, "Mix",
              NormalisableRange<float>(0.f, 1.f, 0.001f), 1.f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"output", 1}, "Output",
              NormalisableRange<float>(-12.f, 12.f, 0.1f), 0.f));
        return l;
    }

    void detect()
    {
        const int N = (int) (0.0175 * sr);
        const int minTau = (int) (sr / 800.0);
        const int maxTau = (int) (sr / 90.0);
        const int total = N + maxTau;
        const int holdHops = juce::jmax(1, (int) std::ceil(0.08 * sr / (double) hop));

        float energy = 0.f;
        for (int j = 0; j < total; ++j)
            scratch[(size_t) j] = det[(size_t) ((wp - total + j) & kMask)];
        for (int j = maxTau; j < total; ++j) energy += scratch[(size_t) j] * scratch[(size_t) j];

        bool voiced = false;
        float midiRaw = 0.f;

        if (std::sqrt(energy / (float) N) >= 0.004f)
        {
            float cum = 0.f;
            yin[0] = 1.f;
            for (int tau = 1; tau <= maxTau; ++tau)
            {
                float d = 0.f;
                for (int j = 0; j < N; j += 2)
                {
                    const float x = scratch[(size_t) (maxTau + j)] - scratch[(size_t) (maxTau + j - tau)];
                    d += x * x;
                }
                cum += d;
                yin[(size_t) tau] = cum > 0.f ? d * (float) tau / cum : 1.f;
            }

            int found = -1;
            for (int tau = juce::jmax(2, minTau); tau < maxTau; ++tau)
            {
                if (yin[(size_t) tau] < 0.15f)
                {
                    while (tau + 1 < maxTau && yin[(size_t) tau + 1] < yin[(size_t) tau]) ++tau;
                    found = tau; break;
                }
            }

            if (found >= 2)
            {
                float tauEst = (float) found;
                const float s0 = yin[(size_t) found - 1], s1 = yin[(size_t) found], s2 = yin[(size_t) found + 1];
                const float den = s0 - 2.f * s1 + s2;
                if (std::abs(den) > 1e-9f) tauEst += 0.5f * (s0 - s2) / den;
                const float f0 = (float) sr / tauEst;
                midiRaw = 69.f + 12.f * std::log2(f0 / 440.f);
                voiced = true;
            }
        }

        if (!voiced)
        {
            if (++unvoicedHops > holdHops)
            {
                // no pitch: pass the voice through untouched
                targetMidiOut = 69.f + 12.f * std::log2(((float) sr / T0cur) / 440.f);
                snapOn = false;
                meterVoiced = false;
                histN = 0; histIdx = 0; curNote = -1;
            }
            return;
        }
        unvoicedHops = 0;

        // median filter over the last detections (removes octave glitches)
        hist[histIdx] = midiRaw;
        histIdx = (histIdx + 1) % 5;
        histN = juce::jmin(5, histN + 1);
        float tmp[5];
        for (int i = 0; i < histN; ++i) tmp[i] = hist[i];
        std::sort(tmp, tmp + histN);
        const float med = tmp[histN / 2];
        const float m = std::abs(midiRaw - med) < 0.7f ? midiRaw : med;

        // choose target note with hysteresis (no flicker between notes)
        const int mask = scaleMask(scaleIdx);
        auto allowed = [&](int note) { return ((mask >> (((note - keyIdx) % 12 + 12) % 12)) & 1) != 0; };

        int cand = (int) std::round(med);
        float bestDist = 1e9f;
        for (int o = -6; o <= 6; ++o)
        {
            const int note = (int) std::round(med) + o;
            if (!allowed(note)) continue;
            const float dist = std::abs((float) note - med);
            if (dist < bestDist) { bestDist = dist; cand = note; }
        }
        if (curNote >= 0 && allowed(curNote) && cand != curNote
            && std::abs(med - (float) cand) + 0.15f > std::abs(med - (float) curNote))
            cand = curNote;
        curNote = cand;
        const float best = (float) curNote;

        // the output pitch is set directly from the target note (exact), not from a ratio
        targetMidiOut = m + (best - m) * amountP;

        // analysis period for the grains
        const float T0meas = (float) sr / (440.f * std::pow(2.f, (med - 69.f) / 12.f));
        T0cur = juce::jlimit((float) (sr / 800.0), (float) (sr / 90.0), T0meas);
        snapOn = true;

        meterDetected = m;
        meterTarget = best;
        meterVoiced = true;
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BlackTune)
};

//==============================================================================
static const juce::Colour kGold(0xfff5c542);
static const juce::Colour kRed(0xffe02020);

class BTLookAndFeel : public juce::LookAndFeel_V4
{
public:
    BTLookAndFeel()
    {
        setColour(juce::PopupMenu::backgroundColourId, juce::Colour(0xff111111));
        setColour(juce::PopupMenu::textColourId, juce::Colours::white);
        setColour(juce::PopupMenu::highlightedBackgroundColourId, kRed);
        setColour(juce::PopupMenu::highlightedTextColourId, juce::Colours::white);
    }

    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float startAngle, float endAngle, juce::Slider&) override
    {
        auto bounds = juce::Rectangle<float>((float) x, (float) y, (float) width, (float) height).reduced(12.f);
        const float r = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.f;
        const auto c = bounds.getCentre();
        const float ang = startAngle + sliderPos * (endAngle - startAngle);

        // scale ticks
        for (int i = 0; i <= 20; ++i)
        {
            const float a = startAngle + (float) i / 20.f * (endAngle - startAngle);
            const float s = std::sin(a), co = std::cos(a);
            const bool major = (i % 5 == 0);
            const float r1 = r + 3.f, r2 = r + (major ? 9.f : 6.f);
            g.setColour(juce::Colours::white.withAlpha(major ? 0.5f : 0.2f));
            g.drawLine(c.x + s * r1, c.y - co * r1, c.x + s * r2, c.y - co * r2, major ? 1.6f : 1.f);
        }

        const float arcR = r - 3.f;
        juce::Path track;
        track.addCentredArc(c.x, c.y, arcR, arcR, 0.f, startAngle, endAngle, true);
        g.setColour(juce::Colour(0xff262626));
        g.strokePath(track, juce::PathStrokeType(4.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        if (sliderPos > 0.002f)
        {
            juce::Path val;
            val.addCentredArc(c.x, c.y, arcR, arcR, 0.f, startAngle, ang, true);
            g.setColour(kGold.withAlpha(0.18f));
            g.strokePath(val, juce::PathStrokeType(10.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            g.setColour(kGold);
            g.strokePath(val, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }

        // metal bezel
        const float bezelR = arcR - 5.f;
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff767676), c.x - bezelR * 0.4f, c.y - bezelR * 0.6f,
                                               juce::Colour(0xff0d0d0d), c.x + bezelR * 0.9f, c.y + bezelR * 0.9f, false));
        g.fillEllipse(c.x - bezelR, c.y - bezelR, bezelR * 2.f, bezelR * 2.f);

        // knurling
        g.setColour(juce::Colours::black.withAlpha(0.55f));
        for (int i = 0; i < 36; ++i)
        {
            const float a = (float) i * juce::MathConstants<float>::twoPi / 36.f;
            const float s = std::sin(a), co = std::cos(a);
            g.drawLine(c.x + s * (bezelR - 2.5f), c.y - co * (bezelR - 2.5f),
                       c.x + s * bezelR, c.y - co * bezelR, 1.f);
        }

        // cap
        const float capR = bezelR - 4.f;
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff535353), c.x - capR * 0.3f, c.y - capR * 0.4f,
                                               juce::Colour(0xff121212), c.x + capR * 0.8f, c.y + capR * 0.9f, true));
        g.fillEllipse(c.x - capR, c.y - capR, capR * 2.f, capR * 2.f);
        g.setColour(juce::Colours::white.withAlpha(0.10f));
        g.fillEllipse(c.x - capR * 0.6f, c.y - capR * 0.85f, capR * 1.0f, capR * 0.5f);

        juce::Path p;
        p.addRoundedRectangle(-1.6f, -capR + 2.f, 3.2f, capR * 0.5f, 1.5f);
        g.setColour(kGold);
        g.fillPath(p, juce::AffineTransform::rotation(ang).translated(c.x, c.y));
    }

    void drawButtonBackground(juce::Graphics& g, juce::Button& b, const juce::Colour&,
                              bool highlighted, bool down) override
    {
        auto r = b.getLocalBounds().toFloat().reduced(0.5f);
        juce::Colour top(0xff2d2d2d), bot(0xff0d0d0d);
        if (highlighted) top = top.brighter(0.25f);
        if (down) std::swap(top, bot);
        g.setGradientFill(juce::ColourGradient(top, 0.f, r.getY(), bot, 0.f, r.getBottom(), false));
        g.fillRoundedRectangle(r, 5.f);
        g.setColour(kGold.withAlpha(highlighted ? 0.95f : 0.45f));
        g.drawRoundedRectangle(r, 5.f, 1.f);
    }

    juce::Font getTextButtonFont(juce::TextButton&, int) override
    {
        return juce::Font(juce::FontOptions(11.f, juce::Font::bold));
    }

    void drawComboBox(juce::Graphics& g, int w, int h, bool, int, int, int, int, juce::ComboBox&) override
    {
        auto r = juce::Rectangle<float>(0.f, 0.f, (float) w, (float) h).reduced(0.5f);
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff262626), 0.f, 0.f,
                                               juce::Colour(0xff0b0b0b), 0.f, (float) h, false));
        g.fillRoundedRectangle(r, 5.f);
        g.setColour(kGold.withAlpha(0.5f));
        g.drawRoundedRectangle(r, 5.f, 1.f);
        juce::Path a;
        a.addTriangle((float) w - 19.f, (float) h * 0.5f - 3.f, (float) w - 9.f, (float) h * 0.5f - 3.f,
                      (float) w - 14.f, (float) h * 0.5f + 3.f);
        g.setColour(kGold);
        g.fillPath(a);
    }
};

//==============================================================================
class BlackTuneEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit BlackTuneEditor(BlackTune& p) : AudioProcessorEditor(&p), proc(p)
    {
        setSize(800, 520);
        setLookAndFeel(&laf);

        const char* ids[6]   = { "retune", "amount", "robot", "grit", "mix", "output" };
        const char* names[6] = { "RETUNE SPEED", "AMOUNT", "ROBOT", "GRIT", "MIX", "OUTPUT" };
        for (int i = 0; i < 6; ++i)
        {
            auto& k = knobs[(size_t) i];
            k.s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
            k.s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 90, 18);
            k.s.setRotaryParameters(juce::MathConstants<float>::pi * 1.2f,
                                    juce::MathConstants<float>::pi * 2.8f, true);
            k.s.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe8e8e8));
            k.s.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
            addAndMakeVisible(k.s);

            k.l.setText(names[i], juce::dontSendNotification);
            k.l.setJustificationType(juce::Justification::centred);
            k.l.setColour(juce::Label::textColourId, i == 0 ? kGold : juce::Colour(0xffbdbdbd));
            k.l.setFont(juce::Font(juce::FontOptions(13.f, juce::Font::bold)));
            addAndMakeVisible(k.l);

            k.a = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, ids[i], k.s);

            if (i == 0)
            {
                k.s.textFromValueFunction = [](double v) { return juce::String(v, 1) + " ms"; };
                k.s.valueFromTextFunction = [](const juce::String& t) { return t.getDoubleValue(); };
            }
            else if (i == 5)
            {
                k.s.textFromValueFunction = [](double v) { return juce::String(v, 1) + " dB"; };
                k.s.valueFromTextFunction = [](const juce::String& t) { return t.getDoubleValue(); };
            }
            else
            {
                k.s.textFromValueFunction = [](double v) { return juce::String((int) std::round(v * 100.0)) + " %"; };
                k.s.valueFromTextFunction = [](const juce::String& t) { return t.getDoubleValue() / 100.0; };
            }
            k.s.updateText();
        }

        setupCombo(keyBox, keyNames());
        setupCombo(scaleBox, scaleNames());
        keyA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, "key", keyBox);
        scaleA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, "scale", scaleBox);

        const char* pn[5] = { "HARD TUNE", "ROBOT", "CLEAN", "NATURAL", "DESTROY" };
        for (int i = 0; i < 5; ++i)
        {
            presets[(size_t) i].setButtonText(pn[i]);
            presets[(size_t) i].setColour(juce::TextButton::textColourOffId, kGold);
            presets[(size_t) i].setColour(juce::TextButton::textColourOnId, kGold);
            presets[(size_t) i].onClick = [this, i] { applyPreset(i); };
            addAndMakeVisible(presets[(size_t) i]);
        }

        startTimerHz(30);
    }

    ~BlackTuneEditor() override { setLookAndFeel(nullptr); }

    void resized() override
    {
        for (int i = 0; i < 6; ++i)
        {
            juce::Rectangle<int> col(20 + i * 127, 378, 127, 130);
            knobs[(size_t) i].l.setBounds(col.removeFromTop(20));
            knobs[(size_t) i].s.setBounds(col);
        }
        keyBox.setBounds(64, 332, 88, 28);
        scaleBox.setBounds(236, 332, 176, 28);
        for (int i = 0; i < 5; ++i)
            presets[(size_t) i].setBounds(434 + i * 68, 332, 64, 28);
    }

    void paint(juce::Graphics& g) override
    {
        const int W = getWidth(), H = getHeight();

        // background: carbon gradient, brushed lines, vignette
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff202020), 0.f, 0.f,
                                               juce::Colour(0xff060606), 0.f, (float) H, false));
        g.fillAll();
        g.setColour(juce::Colours::white.withAlpha(0.018f));
        for (int y = 0; y < H; y += 3) g.drawHorizontalLine(y, 0.f, (float) W);
        g.setGradientFill(juce::ColourGradient(juce::Colours::transparentBlack, (float) W * 0.5f, (float) H * 0.5f,
                                               juce::Colours::black.withAlpha(0.55f), 0.f, 0.f, true));
        g.fillAll();

        drawHeader(g, W);

        drawPanel(g, meterPanel);
        drawPanel(g, kbPanel);
        drawPanel(g, stripPanel);
        drawPanel(g, knobPanel);

        drawMeter(g);
        drawKeyboard(g);

        g.setColour(juce::Colour(0xffbdbdbd));
        g.setFont(juce::Font(juce::FontOptions(13.f, juce::Font::bold)));
        g.drawText("KEY", 32, 332, 32, 28, juce::Justification::centredLeft);
        g.drawText("SCALE", 172, 332, 62, 28, juce::Justification::centredLeft);
    }

private:
    BlackTune& proc;
    BTLookAndFeel laf;

    struct Knob
    {
        juce::Slider s;
        juce::Label l;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> a;
    };
    std::array<Knob, 6> knobs;
    juce::ComboBox keyBox, scaleBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> keyA, scaleA;
    std::array<juce::TextButton, 5> presets;

    const juce::Rectangle<int> meterPanel { 20, 84, 760, 130 };
    const juce::Rectangle<int> kbPanel { 20, 224, 760, 94 };
    const juce::Rectangle<int> stripPanel { 20, 326, 760, 40 };
    const juce::Rectangle<int> knobPanel { 20, 374, 760, 136 };

    void timerCallback() override { repaint(meterPanel.getUnion(kbPanel)); }

    void setupCombo(juce::ComboBox& box, const juce::StringArray& items)
    {
        box.addItemList(items, 1);
        box.setColour(juce::ComboBox::textColourId, juce::Colours::white);
        addAndMakeVisible(box);
    }

    void applyPreset(int idx)
    {
        static const float P[5][5] = {
            { 0.f,  1.f,  0.25f, 0.f,   1.f },   // HARD TUNE
            { 0.f,  1.f,  0.75f, 0.15f, 1.f },   // ROBOT
            { 15.f, 1.f,  0.f,   0.f,   1.f },   // CLEAN
            { 60.f, 0.8f, 0.f,   0.f,   1.f },   // NATURAL
            { 0.f,  1.f,  0.9f,  0.6f,  1.f }    // DESTROY
        };
        const char* ids[5] = { "retune", "amount", "robot", "grit", "mix" };
        for (int i = 0; i < 5; ++i)
            if (auto* prm = proc.apvts.getParameter(ids[i]))
            {
                prm->beginChangeGesture();
                prm->setValueNotifyingHost(prm->convertTo0to1(P[idx][i]));
                prm->endChangeGesture();
            }
    }

    static juce::String noteName(int midiNote)
    {
        return keyNames()[((midiNote % 12) + 12) % 12];
    }

    void drawPanel(juce::Graphics& g, juce::Rectangle<int> r)
    {
        auto f = r.toFloat();
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff1b1b1b), f.getX(), f.getY(),
                                               juce::Colour(0xff090909), f.getX(), f.getBottom(), false));
        g.fillRoundedRectangle(f, 10.f);
        g.setColour(kGold.withAlpha(0.28f));
        g.drawRoundedRectangle(f.reduced(0.5f), 10.f, 1.f);
        g.setColour(juce::Colours::white.withAlpha(0.06f));
        g.drawLine(f.getX() + 10.f, f.getY() + 1.5f, f.getRight() - 10.f, f.getY() + 1.5f);
    }

    void drawHeader(juce::Graphics& g, int W)
    {
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff141414), 0.f, 0.f,
                                               juce::Colour(0xff050505), 0.f, 68.f, false));
        g.fillRect(0, 0, W, 68);

        juce::ColourGradient line(kGold.withAlpha(0.f), 0.f, 0.f, kGold.withAlpha(0.f), (float) W, 0.f, false);
        line.addColour(0.5, kGold);
        g.setGradientFill(line);
        g.fillRect(0, 68, W, 2);

        // vinyl record logo
        const float cx = 46.f, cy = 35.f, R = 24.f;
        g.setColour(kGold.withAlpha(0.25f));
        g.fillEllipse(cx - R - 3.f, cy - R - 3.f, 2.f * R + 6.f, 2.f * R + 6.f);
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff2f2f2f), cx - R * 0.5f, cy - R * 0.5f,
                                               juce::Colour(0xff040404), cx + R, cy + R, true));
        g.fillEllipse(cx - R, cy - R, 2.f * R, 2.f * R);
        for (int i = 1; i <= 5; ++i)
        {
            const float rr = R - 2.f - (float) i * 3.2f;
            g.setColour(juce::Colours::white.withAlpha(0.07f));
            g.drawEllipse(cx - rr, cy - rr, 2.f * rr, 2.f * rr, 1.f);
        }
        g.setColour(kRed);
        g.fillEllipse(cx - 8.f, cy - 8.f, 16.f, 16.f);
        g.setColour(kGold);
        g.fillEllipse(cx - 2.5f, cy - 2.5f, 5.f, 5.f);

        g.setGradientFill(juce::ColourGradient(juce::Colours::white, 0.f, 16.f, kGold, 0.f, 52.f, false));
        g.setFont(juce::Font(juce::FontOptions(32.f, juce::Font::bold)));
        g.drawText("BLACK TUNE RECORD", 84, 0, 480, 68, juce::Justification::centredLeft);

        g.setColour(juce::Colour(0xff9a9a9a));
        g.setFont(juce::Font(juce::FontOptions(12.f, juce::Font::bold)));
        g.drawText("AGGRESSIVE PITCH CORRECTION", W - 320, 0, 300, 68, juce::Justification::centredRight);
    }

    void drawMeter(juce::Graphics& g)
    {
        auto r = meterPanel.toFloat();
        const bool voiced = proc.meterVoiced.load();
        const float det = proc.meterDetected.load();
        const float tgt = proc.meterTarget.load();
        const float devCents = voiced ? juce::jlimit(-50.f, 50.f, (det - tgt) * 100.f) : 0.f;

        // big note with glow
        g.setColour(juce::Colour(0xff777777));
        g.setFont(juce::Font(juce::FontOptions(11.f, juce::Font::bold)));
        g.drawText("TARGET NOTE", (int) r.getX() + 18, (int) r.getY() + 10, 160, 14, juce::Justification::left);

        const juce::String nt = voiced ? noteName((int) std::round(tgt)) : "--";
        g.setFont(juce::Font(juce::FontOptions(70.f, juce::Font::bold)));
        const juce::Rectangle<int> nr((int) r.getX() + 10, (int) r.getY() + 18, 180, 80);
        if (voiced)
        {
            g.setColour(kGold.withAlpha(0.14f));
            for (int dx = -3; dx <= 3; dx += 3)
                for (int dy = -3; dy <= 3; dy += 3)
                    g.drawText(nt, nr.translated(dx, dy), juce::Justification::centred);
        }
        g.setColour(voiced ? kGold : juce::Colour(0xff4a4a4a));
        g.drawText(nt, nr, juce::Justification::centred);

        juce::String status = "NO SIGNAL";
        juce::Colour statusCol(0xff555555);
        if (voiced)
        {
            if (std::abs(devCents) <= 8.f) { status = "LOCKED"; statusCol = juce::Colour(0xff39d353); }
            else if (devCents > 0.f)       { status = "SHARP";  statusCol = kRed; }
            else                           { status = "FLAT";   statusCol = kRed; }
        }
        g.setColour(statusCol);
        g.setFont(juce::Font(juce::FontOptions(12.f, juce::Font::bold)));
        g.drawText(status, (int) r.getX() + 10, (int) r.getY() + 100, 180, 18, juce::Justification::centred);

        // LED tuner bar
        const float bx = r.getX() + 215.f, bw = 460.f, by = r.getY() + 50.f, bh = 32.f;
        const int segs = 41;
        const float segW = bw / (float) segs;
        const int centerIdx = segs / 2;
        const int devIdx = centerIdx + (int) std::round(devCents / 2.5f);
        for (int i = 0; i < segs; ++i)
        {
            const float cents = (float) (i - centerIdx) * 2.5f;
            juce::Colour on = std::abs(cents) <= 10.f ? juce::Colour(0xff39d353)
                            : std::abs(cents) <= 25.f ? kGold : kRed;
            bool lit = false;
            if (voiced)
            {
                const int lo = juce::jmin(centerIdx, devIdx), hi = juce::jmax(centerIdx, devIdx);
                lit = (i >= lo && i <= hi);
            }
            auto seg = juce::Rectangle<float>(bx + (float) i * segW + 1.f, by, segW - 2.f, bh);
            if (lit)
            {
                g.setColour(on.withAlpha(0.22f));
                g.fillRoundedRectangle(seg.expanded(1.5f), 3.f);
                g.setColour(on);
            }
            else
                g.setColour(i == centerIdx ? juce::Colour(0xff3a3a3a) : juce::Colour(0xff191919));
            g.fillRoundedRectangle(seg, 2.f);
        }
        g.setColour(juce::Colour(0xff777777));
        g.setFont(juce::Font(juce::FontOptions(11.f, juce::Font::bold)));
        g.drawText("-50", (int) bx - 10, (int) (by + bh + 6.f), 40, 14, juce::Justification::left);
        g.drawText("0", (int) (bx + bw * 0.5f) - 20, (int) (by + bh + 6.f), 40, 14, juce::Justification::centred);
        g.drawText("+50", (int) (bx + bw) - 30, (int) (by + bh + 6.f), 40, 14, juce::Justification::right);
        g.drawText("CENTS", (int) bx, (int) r.getY() + 12, 100, 14, juce::Justification::left);

        if (voiced)
        {
            g.setColour(juce::Colours::white);
            g.setFont(juce::Font(juce::FontOptions(14.f, juce::Font::bold)));
            g.drawText(juce::String((int) std::round((det - tgt) * 100.f)) + " cents",
                       (int) bx, (int) r.getY() + 26, (int) bw, 18, juce::Justification::centred);
        }

        // output level meter
        const float lvx = r.getRight() - 52.f;
        const float db = juce::Decibels::gainToDecibels(proc.outLevel.load(), -60.f);
        const int lit = (int) std::round(juce::jmap(db, -60.f, 0.f, 0.f, 14.f));
        const float segH = 6.f;
        for (int i = 0; i < 14; ++i)
        {
            juce::Colour on = i < 9 ? juce::Colour(0xff39d353) : i < 12 ? kGold : kRed;
            auto seg = juce::Rectangle<float>(lvx, r.getY() + 18.f + (float) (13 - i) * (segH + 1.5f), 20.f, segH);
            g.setColour(i < lit ? on : juce::Colour(0xff191919));
            g.fillRoundedRectangle(seg, 1.5f);
        }
        g.setColour(juce::Colour(0xff777777));
        g.setFont(juce::Font(juce::FontOptions(11.f, juce::Font::bold)));
        g.drawText("OUT", (int) lvx - 10, (int) r.getBottom() - 22, 40, 14, juce::Justification::centred);
    }

    void drawKeyboard(juce::Graphics& g)
    {
        const int mask = scaleMask((int) *proc.apvts.getRawParameterValue("scale"));
        const int key = (int) *proc.apvts.getRawParameterValue("key");
        const bool voiced = proc.meterVoiced.load();
        const int targetNote = voiced ? (((int) std::round(proc.meterTarget.load())) % 12 + 12) % 12 : -1;

        auto allowed = [&](int note) { return ((mask >> (((note - key) % 12 + 12) % 12)) & 1) != 0; };

        const auto r = kbPanel.reduced(12, 9).toFloat();
        const float w = r.getWidth() / 7.f;
        static const int whites[7] = { 0, 2, 4, 5, 7, 9, 11 };

        for (int i = 0; i < 7; ++i)
        {
            const int note = whites[i];
            auto kr = juce::Rectangle<float>(r.getX() + (float) i * w, r.getY(), w, r.getHeight()).reduced(1.5f);
            const bool isT = (note == targetNote);
            if (isT)
            {
                g.setColour(kGold.withAlpha(0.28f));
                g.fillRoundedRectangle(kr.expanded(3.f), 6.f);
            }
            juce::Colour top = isT ? juce::Colour(0xffffe27a) : allowed(note) ? juce::Colour(0xfff7f7f7) : juce::Colour(0xff3d3d3d);
            juce::Colour bot = isT ? juce::Colour(0xffd49b12) : allowed(note) ? juce::Colour(0xffc4c4c4) : juce::Colour(0xff1d1d1d);
            g.setGradientFill(juce::ColourGradient(top, 0.f, kr.getY(), bot, 0.f, kr.getBottom(), false));
            g.fillRoundedRectangle(kr, 4.f);
            g.setColour(juce::Colours::black.withAlpha(0.6f));
            g.drawRoundedRectangle(kr, 4.f, 1.f);
            g.setColour(isT || allowed(note) ? juce::Colours::black.withAlpha(0.8f) : juce::Colour(0xff8a8a8a));
            g.setFont(juce::Font(juce::FontOptions(12.f, juce::Font::bold)));
            g.drawText(noteName(note), kr.toNearestInt().removeFromBottom(18), juce::Justification::centred);
        }

        static const int blacks[5] = { 1, 3, 6, 8, 10 };
        static const int after[5] = { 0, 1, 3, 4, 5 };
        const float bw = w * 0.6f;
        const float bh = r.getHeight() * 0.58f;
        for (int i = 0; i < 5; ++i)
        {
            const int note = blacks[i];
            auto kr = juce::Rectangle<float>(r.getX() + (float) (after[i] + 1) * w - bw * 0.5f, r.getY(), bw, bh);
            const bool isT = (note == targetNote);
            if (isT)
            {
                g.setColour(kGold.withAlpha(0.3f));
                g.fillRoundedRectangle(kr.expanded(3.f), 5.f);
            }
            juce::Colour top = isT ? juce::Colour(0xffffe27a) : allowed(note) ? juce::Colour(0xffec3535) : juce::Colour(0xff222222);
            juce::Colour bot = isT ? juce::Colour(0xffd49b12) : allowed(note) ? juce::Colour(0xff7d0d0d) : juce::Colour(0xff050505);
            g.setGradientFill(juce::ColourGradient(top, 0.f, kr.getY(), bot, 0.f, kr.getBottom(), false));
            g.fillRoundedRectangle(kr, 3.f);
            g.setColour(juce::Colours::black);
            g.drawRoundedRectangle(kr, 3.f, 1.f);
            g.setColour(juce::Colours::white.withAlpha(0.18f));
            g.drawLine(kr.getX() + 3.f, kr.getY() + 3.f, kr.getRight() - 3.f, kr.getY() + 3.f, 1.f);
        }
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BlackTuneEditor)
};

juce::AudioProcessorEditor* BlackTune::createEditor() { return new BlackTuneEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new BlackTune(); }
