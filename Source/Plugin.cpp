// BLACK TUNE RECORD - aggressive hard-tune / robot vocal effect with custom GUI
#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <atomic>
#include <cmath>
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
        W = (int) std::round(0.023 * sr);
        setLatencySamples(W / 2);

        for (auto& v : dl)   v.assign(kBuf, 0.f);
        for (auto& v : comb) v.assign(kBuf, 0.f);
        det.assign(kBuf, 0.f);
        scratch.assign((size_t) kBuf, 0.f);
        yin.assign((size_t) kBuf, 0.f);

        wp = 0; hopCount = 0; hop = 512;
        phase = 0.f; curRatio = 1.f; targetRatio = 1.f;
        targetFreq = 200.f; combT = (float) (sr / 200.0);
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
        keyIdx   = (int) *apvts.getRawParameterValue("key");
        scaleIdx = (int) *apvts.getRawParameterValue("scale");
        amountP = amount;

        const float coef = retMs <= 0.01f ? 1.f
                         : 1.f - std::exp(-1.f / (retMs * 0.001f * (float) sr));
        const float drive = 1.f + grit * 9.f;
        const float levels = std::pow(2.f, 16.f - grit * 11.f);
        const float fb = robot * 0.85f;
        const float combNorm = 1.f / (1.f + fb * 0.6f);

        for (int i = 0; i < n; ++i)
        {
            float mono = 0.f;
            for (int c = 0; c < nCh; ++c) mono += buf.getReadPointer(c)[i];
            mono /= (float) nCh;

            det[(size_t) (wp & kMask)] = mono;
            if (++hopCount >= hop) { hopCount = 0; detect(); }

            curRatio += coef * (targetRatio - curRatio);
            combT    += 0.002f * ((float) (sr / targetFreq) - combT);
            combT = juce::jlimit(8.f, (float) (kBuf - 4), combT);

            phase += (1.f - curRatio) / (float) W;
            phase -= std::floor(phase);
            float ph2 = phase + 0.5f; ph2 -= std::floor(ph2);
            const float g1 = 0.5f - 0.5f * std::cos(juce::MathConstants<float>::twoPi * phase);
            const float g2 = 0.5f - 0.5f * std::cos(juce::MathConstants<float>::twoPi * ph2);
            const float d1 = phase * (float) W + 1.f;
            const float d2 = ph2 * (float) W + 1.f;

            for (int c = 0; c < nCh; ++c)
            {
                float* io = buf.getWritePointer(c);
                const float in = io[i];
                dl[(size_t) c][(size_t) (wp & kMask)] = in;

                float wet = g1 * readFrac(dl[(size_t) c], d1) + g2 * readFrac(dl[(size_t) c], d2);

                float yc = wet + fb * readFrac(comb[(size_t) c], combT);
                comb[(size_t) c][(size_t) (wp & kMask)] = yc;
                wet = yc * combNorm;

                wet = std::tanh(wet * drive);
                wet = std::round(wet * levels) / levels;

                const float dry = dl[(size_t) c][(size_t) ((wp - W / 2) & kMask)];
                io[i] = dry * (1.f - mix) + wet * mix;
            }
            ++wp;
        }
    }

    // shared with the editor
    juce::AudioProcessorValueTreeState apvts;
    std::atomic<float> meterDetected { 0.f };
    std::atomic<float> meterTarget { 0.f };
    std::atomic<bool> meterVoiced { false };

private:
    static constexpr int kBuf = 1 << 14;
    static constexpr int kMask = kBuf - 1;

    double sr = 44100.0;
    int W = 1024, wp = 0, hopCount = 0, hop = 512;
    float phase = 0.f, curRatio = 1.f, targetRatio = 1.f, targetFreq = 200.f, combT = 200.f;
    int keyIdx = 0, scaleIdx = 0;
    float amountP = 1.f;
    std::array<std::vector<float>, 2> dl, comb;
    std::vector<float> det, scratch, yin;

    float readFrac(const std::vector<float>& b, float delay) const
    {
        const float pos = (float) wp - delay;
        const int i0 = (int) std::floor(pos);
        const float f = pos - (float) i0;
        const float a = b[(size_t) (i0 & kMask)];
        const float c = b[(size_t) ((i0 + 1) & kMask)];
        return a + f * (c - a);
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
              NormalisableRange<float>(0.f, 1.f, 0.001f), 0.5f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"grit", 1}, "Grit",
              NormalisableRange<float>(0.f, 1.f, 0.001f), 0.3f));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"mix", 1}, "Mix",
              NormalisableRange<float>(0.f, 1.f, 0.001f), 1.f));
        return l;
    }

    void detect()
    {
        const int N = (int) (0.0175 * sr);
        const int minTau = (int) (sr / 800.0);
        const int maxTau = (int) (sr / 70.0);
        const int total = N + maxTau;

        float energy = 0.f;
        for (int j = 0; j < total; ++j)
            scratch[(size_t) j] = det[(size_t) ((wp - total + j) & kMask)];
        for (int j = maxTau; j < total; ++j) energy += scratch[(size_t) j] * scratch[(size_t) j];
        if (std::sqrt(energy / (float) N) < 0.004f) { targetRatio = 1.f; meterVoiced = false; return; }

        float cum = 0.f;
        yin[0] = 1.f;
        for (int tau = 1; tau <= maxTau; ++tau)
        {
            float d = 0.f;
            for (int j = 0; j < N; ++j)
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
        if (found < 0) { targetRatio = 1.f; meterVoiced = false; return; }

        float tauEst = (float) found;
        const float s0 = yin[(size_t) found - 1], s1 = yin[(size_t) found], s2 = yin[(size_t) found + 1];
        const float den = s0 - 2.f * s1 + s2;
        if (std::abs(den) > 1e-9f) tauEst += 0.5f * (s0 - s2) / den;

        const float f0 = (float) sr / tauEst;
        const float midi = 69.f + 12.f * std::log2(f0 / 440.f);

        const int mask = scaleMask(scaleIdx);
        float best = midi; float bestDist = 1e9f;
        const int base = (int) std::round(midi);
        for (int o = -6; o <= 6; ++o)
        {
            const int note = base + o;
            const int rel = ((note - keyIdx) % 12 + 12) % 12;
            if (!((mask >> rel) & 1)) continue;
            const float dist = std::abs((float) note - midi);
            if (dist < bestDist) { bestDist = dist; best = (float) note; }
        }

        targetRatio = std::pow(2.f, ((best - midi) * amountP) / 12.f);
        targetFreq = 440.f * std::pow(2.f, ((midi + (best - midi) * amountP) - 69.f) / 12.f);

        meterDetected = midi;
        meterTarget = best;
        meterVoiced = true;
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BlackTune)
};

//==============================================================================
// Look and feel: black metal knobs with a red value arc and gold pointer
class BTLookAndFeel : public juce::LookAndFeel_V4
{
public:
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float startAngle, float endAngle, juce::Slider&) override
    {
        auto bounds = juce::Rectangle<float>((float) x, (float) y, (float) width, (float) height).reduced(8.f);
        const float r = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.f;
        const auto c = bounds.getCentre();
        const float ang = startAngle + sliderPos * (endAngle - startAngle);

        juce::Path track;
        track.addCentredArc(c.x, c.y, r, r, 0.f, startAngle, endAngle, true);
        g.setColour(juce::Colour(0xff2a2a2a));
        g.strokePath(track, juce::PathStrokeType(5.f));

        juce::Path val;
        val.addCentredArc(c.x, c.y, r, r, 0.f, startAngle, ang, true);
        g.setColour(juce::Colour(0xffe02020));
        g.strokePath(val, juce::PathStrokeType(5.f, juce::PathStrokeType::curved,
                                               juce::PathStrokeType::rounded));

        const float br = r - 9.f;
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff555555), c.x - br * 0.5f, c.y - br * 0.7f,
                                               juce::Colour(0xff0c0c0c), c.x + br * 0.6f, c.y + br * 0.8f, false));
        g.fillEllipse(c.x - br, c.y - br, br * 2.f, br * 2.f);
        g.setColour(juce::Colours::black);
        g.drawEllipse(c.x - br, c.y - br, br * 2.f, br * 2.f, 2.f);

        juce::Path p;
        p.addRoundedRectangle(-2.f, -br + 3.f, 4.f, br * 0.45f, 1.5f);
        g.setColour(juce::Colour(0xfff5c542));
        g.fillPath(p, juce::AffineTransform::rotation(ang).translated(c.x, c.y));
    }
};

//==============================================================================
class BlackTuneEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit BlackTuneEditor(BlackTune& p) : AudioProcessorEditor(&p), proc(p)
    {
        setSize(760, 480);
        setLookAndFeel(&laf);

        const char* ids[5]   = { "retune", "amount", "robot", "grit", "mix" };
        const char* names[5] = { "RETUNE SPEED", "AMOUNT", "ROBOT", "GRIT", "MIX" };
        for (int i = 0; i < 5; ++i)
        {
            auto& k = knobs[(size_t) i];
            k.s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
            k.s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 18);
            k.s.setRotaryParameters(juce::MathConstants<float>::pi * 1.2f,
                                    juce::MathConstants<float>::pi * 2.8f, true);
            k.s.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
            k.s.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
            if (i == 0) k.s.setTextValueSuffix(" ms");
            addAndMakeVisible(k.s);

            k.l.setText(names[i], juce::dontSendNotification);
            k.l.setJustificationType(juce::Justification::centred);
            k.l.setColour(juce::Label::textColourId, i == 0 ? juce::Colour(0xffe02020) : juce::Colour(0xffcccccc));
            k.l.setFont(juce::Font(juce::FontOptions(14.f, juce::Font::bold)));
            addAndMakeVisible(k.l);

            k.a = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, ids[i], k.s);
        }

        setupCombo(keyBox, keyNames());
        setupCombo(scaleBox, scaleNames());
        keyA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, "key", keyBox);
        scaleA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, "scale", scaleBox);

        startTimerHz(30);
    }

    ~BlackTuneEditor() override { setLookAndFeel(nullptr); }

    void resized() override
    {
        for (int i = 0; i < 5; ++i)
        {
            juce::Rectangle<int> col(20 + i * 144, 330, 144, 140);
            knobs[(size_t) i].l.setBounds(col.removeFromTop(18));
            knobs[(size_t) i].s.setBounds(col);
        }
        keyBox.setBounds(60, 290, 90, 28);
        scaleBox.setBounds(235, 290, 180, 28);
    }

    void paint(juce::Graphics& g) override
    {
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff1d1d1d), 0.f, 0.f,
                                               juce::Colour(0xff050505), 0.f, (float) getHeight(), false));
        g.fillAll();

        // header
        g.setColour(juce::Colour(0xff0a0a0a));
        g.fillRect(0, 0, getWidth(), 64);
        g.setColour(juce::Colour(0xffe02020));
        g.fillRect(0, 62, getWidth(), 2);
        g.setColour(juce::Colours::white);
        g.setFont(juce::Font(juce::FontOptions(34.f, juce::Font::bold)));
        g.drawText("BLACK TUNE RECORD", 20, 0, 520, 64, juce::Justification::centredLeft);
        g.setColour(juce::Colour(0xff999999));
        g.setFont(juce::Font(juce::FontOptions(13.f, juce::Font::plain)));
        g.drawText("AGGRESSIVE PITCH CORRECTION", 440, 0, 300, 64, juce::Justification::centredRight);

        drawMeter(g);
        drawKeyboard(g);

        g.setColour(juce::Colour(0xffcccccc));
        g.setFont(juce::Font(juce::FontOptions(14.f, juce::Font::bold)));
        g.drawText("KEY", 20, 290, 40, 28, juce::Justification::centredLeft);
        g.drawText("SCALE", 170, 290, 60, 28, juce::Justification::centredLeft);
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
    std::array<Knob, 5> knobs;
    juce::ComboBox keyBox, scaleBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> keyA, scaleA;

    const juce::Rectangle<int> meterArea { 20, 76, 720, 110 };
    const juce::Rectangle<int> kbArea { 20, 198, 720, 80 };

    void timerCallback() override { repaint(meterArea.getUnion(kbArea)); }

    void setupCombo(juce::ComboBox& box, const juce::StringArray& items)
    {
        box.addItemList(items, 1);
        box.setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff151515));
        box.setColour(juce::ComboBox::textColourId, juce::Colours::white);
        box.setColour(juce::ComboBox::outlineColourId, juce::Colour(0xffe02020));
        box.setColour(juce::ComboBox::arrowColourId, juce::Colour(0xfff5c542));
        addAndMakeVisible(box);
    }

    static juce::String noteName(int midiNote)
    {
        return keyNames()[((midiNote % 12) + 12) % 12];
    }

    void drawMeter(juce::Graphics& g)
    {
        auto r = meterArea.toFloat();
        g.setColour(juce::Colour(0xff0d0d0d));
        g.fillRoundedRectangle(r, 8.f);
        g.setColour(juce::Colour(0xff333333));
        g.drawRoundedRectangle(r, 8.f, 1.5f);

        const bool voiced = proc.meterVoiced.load();
        const float det = proc.meterDetected.load();
        const float tgt = proc.meterTarget.load();

        g.setColour(voiced ? juce::Colour(0xfff5c542) : juce::Colour(0xff555555));
        g.setFont(juce::Font(juce::FontOptions(64.f, juce::Font::bold)));
        g.drawText(voiced ? noteName((int) std::round(tgt)) : "--",
                   (int) r.getX() + 10, (int) r.getY(), 170, (int) r.getHeight(),
                   juce::Justification::centred);

        auto bar = juce::Rectangle<float>(220.f, r.getCentreY() - 14.f, 480.f, 28.f);
        g.setColour(juce::Colour(0xff050505));
        g.fillRoundedRectangle(bar, 4.f);
        g.setColour(juce::Colour(0xff444444));
        g.drawRoundedRectangle(bar, 4.f, 1.f);

        for (int cents = -50; cents <= 50; cents += 25)
        {
            const float x = bar.getCentreX() + (float) cents / 50.f * bar.getWidth() * 0.5f;
            g.setColour(cents == 0 ? juce::Colour(0xff39d353) : juce::Colour(0xff555555));
            g.drawLine(x, bar.getY() + 3.f, x, bar.getBottom() - 3.f, cents == 0 ? 2.f : 1.f);
        }

        g.setColour(juce::Colour(0xff888888));
        g.setFont(juce::Font(juce::FontOptions(11.f, juce::Font::plain)));
        g.drawText("FLAT", (int) bar.getX(), (int) bar.getBottom() + 4, 60, 14, juce::Justification::left);
        g.drawText("SHARP", (int) bar.getRight() - 60, (int) bar.getBottom() + 4, 60, 14, juce::Justification::right);

        if (voiced)
        {
            const float dev = juce::jlimit(-50.f, 50.f, (det - tgt) * 100.f);
            const float x = bar.getCentreX() + dev / 50.f * bar.getWidth() * 0.5f;
            g.setColour(juce::Colour(0xffe02020));
            g.fillRoundedRectangle(x - 3.f, bar.getY() - 5.f, 6.f, bar.getHeight() + 10.f, 2.f);
            g.setColour(juce::Colours::white);
            g.setFont(juce::Font(juce::FontOptions(13.f, juce::Font::bold)));
            g.drawText(juce::String((int) std::round((det - tgt) * 100.f)) + " cents",
                       (int) bar.getX(), (int) bar.getY() - 22, (int) bar.getWidth(), 16,
                       juce::Justification::centred);
        }
    }

    void drawKeyboard(juce::Graphics& g)
    {
        const int mask = scaleMask((int) *proc.apvts.getRawParameterValue("scale"));
        const int key = (int) *proc.apvts.getRawParameterValue("key");
        const bool voiced = proc.meterVoiced.load();
        const int targetNote = voiced ? (((int) std::round(proc.meterTarget.load())) % 12 + 12) % 12 : -1;

        auto allowed = [&](int note) { return ((mask >> (((note - key) % 12 + 12) % 12)) & 1) != 0; };

        const auto r = kbArea.toFloat();
        const float w = r.getWidth() / 7.f;
        static const int whites[7] = { 0, 2, 4, 5, 7, 9, 11 };

        for (int i = 0; i < 7; ++i)
        {
            const int note = whites[i];
            auto kr = juce::Rectangle<float>(r.getX() + (float) i * w, r.getY(), w, r.getHeight()).reduced(1.5f);
            g.setColour(note == targetNote ? juce::Colour(0xfff5c542)
                        : allowed(note) ? juce::Colour(0xffeeeeee) : juce::Colour(0xff3a3a3a));
            g.fillRoundedRectangle(kr, 3.f);
            g.setColour(note == targetNote || allowed(note) ? juce::Colours::black : juce::Colour(0xff888888));
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
            g.setColour(note == targetNote ? juce::Colour(0xfff5c542)
                        : allowed(note) ? juce::Colour(0xffd01818) : juce::Colour(0xff0b0b0b));
            g.fillRoundedRectangle(kr, 3.f);
            g.setColour(juce::Colour(0xff000000));
            g.drawRoundedRectangle(kr, 3.f, 1.f);
        }
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BlackTuneEditor)
};

juce::AudioProcessorEditor* BlackTune::createEditor() { return new BlackTuneEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new BlackTune(); }
