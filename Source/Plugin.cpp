// BLACK TUNE RECORD v6 (look-ahead note tracking, pitch graph, premium UI)
// Three independent tabs in one plugin: AUTO TUNE / FRESH AIR / EQUALIZER
// - Auto Tune: YIN tracking + median + hysteresis, PSOLA engine with exact window normalisation
// - Fresh Air: dynamic high-band lift + harmonic exciter with sibilance protection
// - EQ: low cut, 4 bands with adjustable frequency, high cut, live response curve
#include <juce_audio_processors/juce_audio_processors.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
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

// parameter ids of the EQ and Fresh Air modules (order matters: used by processor and editor)
static const char* const kEqIds[10] = { "eq_lowcut", "eq_low", "eq_low_f", "eq_lomid", "eq_lomid_f",
                                        "eq_himid", "eq_himid_f", "eq_high", "eq_high_f", "eq_highcut" };
static const char* const kAirIds[4] = { "air_amount", "air_freq", "air_sparkle", "air_smooth" };

//==============================================================================
// RBJ biquad (double precision, per-channel state)
struct Biquad
{
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double z1[2] = { 0, 0 }, z2[2] = { 0, 0 };

    void reset() { z1[0] = z1[1] = z2[0] = z2[1] = 0.0; }

    float process(int ch, float xf)
    {
        const double x = (double) xf;
        const double y = b0 * x + z1[ch];
        z1[ch] = b1 * x - a1 * y + z2[ch];
        z2[ch] = b2 * x - a2 * y;
        return (float) y;
    }

    void set(double B0, double B1, double B2, double A0, double A1, double A2)
    {
        b0 = B0 / A0; b1 = B1 / A0; b2 = B2 / A0; a1 = A1 / A0; a2 = A2 / A0;
    }

    static double clampF(double sr, double f) { return std::min(std::max(f, 10.0), 0.45 * sr); }

    void setHighPass(double sr, double f, double Q)
    {
        const double w = 6.283185307179586 * clampF(sr, f) / sr, c = std::cos(w), al = std::sin(w) / (2.0 * Q);
        set((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al);
    }
    void setLowPass(double sr, double f, double Q)
    {
        const double w = 6.283185307179586 * clampF(sr, f) / sr, c = std::cos(w), al = std::sin(w) / (2.0 * Q);
        set((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al);
    }
    void setPeak(double sr, double f, double Q, double dB)
    {
        const double A = std::pow(10.0, dB / 40.0);
        const double w = 6.283185307179586 * clampF(sr, f) / sr, c = std::cos(w), al = std::sin(w) / (2.0 * Q);
        set(1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A);
    }
    void setLowShelf(double sr, double f, double dB)
    {
        const double A = std::pow(10.0, dB / 40.0);
        const double w = 6.283185307179586 * clampF(sr, f) / sr, c = std::cos(w), s = std::sin(w);
        const double beta = 2.0 * std::sqrt(A) * (s / 2.0 * std::sqrt(2.0));
        set(A * ((A + 1) - (A - 1) * c + beta), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - beta),
            (A + 1) + (A - 1) * c + beta, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - beta);
    }
    void setHighShelf(double sr, double f, double dB)
    {
        const double A = std::pow(10.0, dB / 40.0);
        const double w = 6.283185307179586 * clampF(sr, f) / sr, c = std::cos(w), s = std::sin(w);
        const double beta = 2.0 * std::sqrt(A) * (s / 2.0 * std::sqrt(2.0));
        set(A * ((A + 1) + (A - 1) * c + beta), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - beta),
            (A + 1) - (A - 1) * c + beta, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - beta);
    }

    // complex frequency response at angular frequency w (radians/sample), used by the editor curves
    std::complex<double> response(double w) const
    {
        const std::complex<double> e1 = std::polar(1.0, -w), e2 = std::polar(1.0, -2.0 * w);
        return (b0 + b1 * e1 + b2 * e2) / (1.0 + a1 * e1 + a2 * e2);
    }
};

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
        for (auto& a : trailDet) a.store(-1.f);
        for (auto& a : trailOut) a.store(-1.f);
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
        wsum.assign(kBuf, 0.f);
        det.assign(kBuf, 0.f);
        lp.assign(kBuf, 0.f);
        scratch.assign((size_t) kBuf, 0.f);
        yin.assign((size_t) kBuf, 0.f);

        wp = 0; hopCount = 0;
        hop = juce::jmax(64, (int) (0.0058 * sr));
        T0cur = (float) (sr / 200.0);
        curMidiOut = targetMidiOut = 69.f + 12.f * std::log2(200.f / 440.f);
        outFreq = 200.f;
        nextTs = sr / 90.0 + 8.0;                     // first grain never reaches negative time
        anaMark = nextTs - (double) delay0;
        combT = (float) (sr / 200.0);
        combLP[0] = combLP[1] = 0.f;
        lpState = 0.f; detLP = 0.f;
        snapOn = false;
        trkN = 0; lastDetVoiced = false; lastDetMidi = 60.f;
        curNote = -1; resVoiced = false; resM = 60.f;
        trailCnt = 0; trailEvery = juce::jmax(1, (int) std::round(0.012 * sr));
        meterVoiced = false;

        for (Biquad* b : { &eqHP, &eqLow, &eqLoMid, &eqHiMid, &eqHigh, &eqLP, &airHP, &airLP, &airOutHP, &airShelf }) b->reset();
        airEnv[0] = airEnv[1] = 0.f;
        prevEqOn = prevAirOn = false;
        eqFade = airFade = 0.f;
        tuneFade = apvts.getRawParameterValue("tune_on")->load() > 0.5f ? 1.f : 0.f;
        fxInit = false;
    }
    void releaseResources() override {}

    void processBlock(juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int nCh = juce::jmin(2, buf.getNumChannels());
        const int n = buf.getNumSamples();

        auto P = [this](const char* id) { return apvts.getRawParameterValue(id)->load(); };

        // ---- AUTO TUNE parameters ----
        const bool tuneOn = P("tune_on") > 0.5f;
        const float amount = P("amount");
        const float retMs  = P("retune");
        const float robot  = P("robot");
        const float grit   = P("grit");
        const float mix    = P("mix");
        const float outDb  = P("output");
        keyIdx   = (int) P("key");
        scaleIdx = (int) P("scale");
        amountP = amount;
        trackAmt = P("track");

        // ---- EQ / FRESH AIR parameters (smoothed per block: no zipper noise) ----
        const bool eqOn  = P("eq_on") > 0.5f;
        const bool airOn = P("air_on") > 0.5f;
        float eqT[10], airT[4];
        for (int i = 0; i < 10; ++i) eqT[i] = P(kEqIds[i]);
        for (int i = 0; i < 4; ++i)  airT[i] = P(kAirIds[i]);
        if (!fxInit)
        {
            for (int i = 0; i < 10; ++i) eqS[i] = eqT[i];
            for (int i = 0; i < 4; ++i)  airS[i] = airT[i];
            fxInit = true;
        }
        else
        {
            const float sm = 1.f - std::exp(-(float) n / (0.03f * (float) sr));
            for (int i = 0; i < 10; ++i) eqS[i] += sm * (eqT[i] - eqS[i]);
            for (int i = 0; i < 4; ++i)  airS[i] += sm * (airT[i] - airS[i]);
        }

        if (eqOn && !prevEqOn)
            for (Biquad* b : { &eqHP, &eqLow, &eqLoMid, &eqHiMid, &eqHigh, &eqLP }) b->reset();
        if (airOn && !prevAirOn)
        {
            for (Biquad* b : { &airHP, &airLP, &airOutHP, &airShelf }) b->reset();
            airEnv[0] = airEnv[1] = 0.f;
        }
        const bool eqUseHP = eqS[0] > 20.5f;
        const bool eqUseLP = eqS[9] < 19900.f;
        if (eqOn || eqFade > 0.0005f)
        {
            eqHP.setHighPass(sr, (double) eqS[0], 0.7071);
            eqLow.setLowShelf(sr, (double) eqS[2], (double) eqS[1]);
            eqLoMid.setPeak(sr, (double) eqS[4], 0.9, (double) eqS[3]);
            eqHiMid.setPeak(sr, (double) eqS[6], 0.9, (double) eqS[5]);
            eqHigh.setHighShelf(sr, (double) eqS[8], (double) eqS[7]);
            eqLP.setLowPass(sr, (double) eqS[9], 0.7071);
        }
        const float sparkle = airS[2];
        const float smoothAmt = airS[3];
        if (airOn || airFade > 0.0005f)
        {
            airHP.setHighPass(sr, (double) airS[1], 0.7071);
            airOutHP.setHighPass(sr, (double) airS[1], 0.7071);
            airLP.setLowPass(sr, std::min(12000.0, 0.4 * sr), 0.7071);
            airShelf.setHighShelf(sr, (double) airS[1] * 0.8, (double) airS[0] * 8.0);   // up to +8 dB
        }
        prevEqOn = eqOn;
        prevAirOn = airOn;

        const float minCoef = 1.f - std::exp(-1.f / (0.0015f * (float) sr));
        const float coef = retMs <= 0.01f ? minCoef
                         : juce::jmax(minCoef, 1.f - std::exp(-1.f / (retMs * 0.001f * (float) sr)));
        const float releaseCoef = 1.f - std::exp(-1.f / (0.02f * (float) sr));
        const float lpA  = 1.f - std::exp(-juce::MathConstants<float>::twoPi * 900.f / (float) sr);
        const float detA = 1.f - std::exp(-juce::MathConstants<float>::twoPi * 80.f / (float) sr);
        const float fadeCoef = 1.f - std::exp(-1.f / (0.005f * (float) sr));
        const float atkCoef = 1.f - std::exp(-1.f / (0.001f * (float) sr));
        const float relCoef = 1.f - std::exp(-1.f / (0.04f * (float) sr));

        const float drive = 1.f + grit * 6.f;
        const float invDrive = 1.f / std::sqrt(drive);
        const float levels = std::pow(2.f, 16.f - grit * 8.f);
        const float fb = robot * 0.7f;
        const float combNorm = 1.f / (1.f + fb * 0.6f);
        const float outGain = juce::Decibels::decibelsToGain(outDb);

        float pk = 0.f, airAcc = 0.f;

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
            detLP += detA * (mono - detLP);                  // 80 Hz high-pass for the pitch detector
            det[(size_t) (wp & kMask)] = mono - detLP;
            lpState += lpA * (mono - lpState);
            lp[(size_t) (wp & kMask)] = lpState;

            if (++hopCount >= hop) { hopCount = 0; detect(); }

            curMidiOut += (resVoiced ? coef : releaseCoef) * (targetMidiOut - curMidiOut);
            curMidiOut = juce::jlimit(20.f, 120.f, curMidiOut);

            generateGrains(nCh);

            if (++trailCnt >= trailEvery)   // history for the pitch graph
            {
                trailCnt = 0;
                const int tp = trailPos.load();
                trailDet[(size_t) tp].store(resVoiced ? resM : -1.f);
                trailOut[(size_t) tp].store(resVoiced ? curMidiOut : -1.f);
                trailPos.store((tp + 1) % kTrail);
            }

            combT += 0.002f * ((float) (sr / (double) outFreq) - combT);
            combT = juce::jlimit(8.f, (float) (kBuf - 4), combT);

            eqFade   += fadeCoef * ((eqOn   ? 1.f : 0.f) - eqFade);
            airFade  += fadeCoef * ((airOn  ? 1.f : 0.f) - airFade);
            tuneFade += fadeCoef * ((tuneOn ? 1.f : 0.f) - tuneFade);
            const float mixEff = mix * tuneFade;

            // exact overlap-add normalisation: constant level whatever the shift
            const size_t wi = (size_t) (wp & kMask);
            const float norm = 1.f / juce::jmax(wsum[wi], 0.2f);
            wsum[wi] = 0.f;

            for (int c = 0; c < nCh; ++c)
            {
                float* io = buf.getWritePointer(c);

                float wet = outB[(size_t) c][wi] * norm;
                outB[(size_t) c][wi] = 0.f;

                // robot comb with a damped feedback path (cleaner, less harsh ringing)
                const float fbRead = readAt(comb[(size_t) c], (double) wp - (double) combT);
                combLP[c] += 0.55f * (fbRead - combLP[c]);
                const float yc = wet + fb * combLP[c];
                comb[(size_t) c][wi] = yc;
                wet = yc * combNorm;

                if (grit > 0.001f)
                {
                    wet = std::tanh(wet * drive) * invDrive;
                    wet = std::round(wet * levels) / levels;
                }

                const float dry = dl[(size_t) c][(size_t) ((wp - delay0) & kMask)];
                float out = dry * (1.f - mixEff) + wet * mixEff;

                // ---- EQUALIZER ----
                if (eqFade > 0.0005f)
                {
                    float y = out;
                    if (eqUseHP) y = eqHP.process(c, y);
                    y = eqLow.process(c, y);
                    y = eqLoMid.process(c, y);
                    y = eqHiMid.process(c, y);
                    y = eqHigh.process(c, y);
                    if (eqUseLP) y = eqLP.process(c, y);
                    out += eqFade * (y - out);
                }

                // ---- FRESH AIR: dynamic lift + harmonic exciter, tamed on sibilance ----
                if (airFade > 0.0005f)
                {
                    const float band = airHP.process(c, out);
                    const float a = std::abs(band);
                    airEnv[c] += (a > airEnv[c] ? atkCoef : relCoef) * (a - airEnv[c]);
                    const float e = airEnv[c] + 1e-7f;
                    const float dyn = 1.f / (1.f + smoothAmt * 30.f * airEnv[c]);

                    // smooth shelf lift (no phase dip below the corner), reduced when the highs are loud
                    const float lift = (airShelf.process(c, out) - out) * dyn;

                    const float bl = airLP.process(c, band);
                    const float sat = e * std::tanh(1.2f * bl / e);
                    const float exc = airOutHP.process(c, 0.5f * sat + (std::abs(bl) - e));

                    const float add = (lift + exc * sparkle * 0.6f * dyn) * airFade;
                    out += add;
                    airAcc += std::abs(add);
                }

                out = softClip(out * outGain);
                io[i] = out;
                pk = juce::jmax(pk, std::abs(out));
            }
            ++wp;
        }
        outLevel.store(juce::jmax(pk, outLevel.load() * 0.9f));
        {
            const float lvl = airAcc / (float) (juce::jmax(1, n) * juce::jmax(1, nCh));
            const float db = 20.f * std::log10(lvl + 1e-5f);
            airActivity.store(juce::jmax(juce::jlimit(0.f, 1.f, (db + 70.f) / 45.f), airActivity.load() * 0.85f));
        }
    }

    // shared with the editor
    juce::AudioProcessorValueTreeState apvts;
    std::atomic<float> meterDetected { 0.f };
    std::atomic<float> meterTarget { 0.f };
    std::atomic<bool> meterVoiced { false };
    std::atomic<float> outLevel { 0.f };
    std::atomic<float> airActivity { 0.f };

    // short pitch history (input pitch and corrected pitch) for the graph in the editor
    static constexpr int kTrail = 256;
    std::array<std::atomic<float>, kTrail> trailDet, trailOut;
    std::atomic<int> trailPos { 0 };

private:
    static constexpr int kBuf = 1 << 15;
    static constexpr int kMask = kBuf - 1;

    double sr = 44100.0;
    int delay0 = 1500, hopCount = 0, hop = 256;
    long long wp = 0;
    double nextTs = 0.0, anaMark = 0.0;
    float T0cur = 220.f, curMidiOut = 55.f, targetMidiOut = 55.f, outFreq = 200.f, combT = 200.f;
    float lpState = 0.f, detLP = 0.f;
    float combLP[2] = { 0.f, 0.f };
    bool snapOn = false;
    int keyIdx = 0, scaleIdx = 0;
    float amountP = 1.f;
    // pitch track: one measurement per detection hop, stamped with the input time it describes
    struct TrackPt { double t = 0; float midi = 0, conf = 0, T0 = 0; };
    static constexpr int kTrk = 1024;
    std::array<TrackPt, kTrk> trk;
    long long trkN = 0;
    bool lastDetVoiced = false;
    float lastDetMidi = 60.f;
    int curNote = -1;
    bool resVoiced = false;
    float resM = 60.f, trackAmt = 0.5f;
    int trailCnt = 0, trailEvery = 530;
    std::array<std::vector<float>, 2> dl, outB, comb;
    std::vector<float> wsum, det, lp, scratch, yin;

    Biquad eqHP, eqLow, eqLoMid, eqHiMid, eqHigh, eqLP, airHP, airLP, airOutHP, airShelf;
    float airEnv[2] = { 0.f, 0.f };
    bool prevEqOn = false, prevAirOn = false, fxInit = false;
    float eqFade = 0.f, airFade = 0.f, tuneFade = 1.f;   // smooth on/off so toggling never clicks
    float eqS[10] = { 0 }, airS[4] = { 0 };

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

    // Decide the target note for the audio that is about to be synthesised (input time tdes).
    // The engine has ~33 ms of latency, so the pitch measurements AFTER tdes already exist:
    // the note is voted from the measurements before and after that moment (no lag, no wrong
    // notes when the singer changes note fast). Glides and glitches lose the vote.
    void resolveTarget(double tdes)
    {
        const double wb = (0.005 + 0.011 * (double) trackAmt) * sr;   // look back  5..16 ms
        const double wa = (0.006 + 0.012 * (double) trackAmt) * sr;   // look ahead 6..18 ms
        const int mask = scaleMask(scaleIdx);
        auto allowed = [&](int note) { return ((mask >> (((note - keyIdx) % 12 + 12) % 12)) & 1) != 0; };

        int noteIds[16]; float votes[16]; int nNotes = 0;
        float midis[16]; double times[16]; float T0s[16]; int nPts = 0;

        for (long long k = trkN - 1; k >= 0 && k > trkN - kTrk + 1 && nPts < 16; --k)
        {
            const TrackPt& p = trk[(size_t) (k % kTrk)];
            if (p.t < tdes - wb) break;
            if (p.t > tdes + wa) continue;

            midis[nPts] = p.midi; times[nPts] = p.t; T0s[nPts] = p.T0; ++nPts;

            const double dist = std::abs(p.t - tdes) / (p.t < tdes ? wb : wa);
            const float w = p.conf * (1.f - 0.5f * (float) dist);
            const int near = (int) std::round(p.midi);
            int bn = near; float bd = 1e9f;
            for (int o = -6; o <= 6; ++o)
            {
                if (!allowed(near + o)) continue;
                const float d = std::abs((float) (near + o) - p.midi);
                if (d < bd) { bd = d; bn = near + o; }
            }
            int slot = -1;
            for (int i = 0; i < nNotes; ++i) if (noteIds[i] == bn) slot = i;
            if (slot < 0 && nNotes < 16) { slot = nNotes++; noteIds[slot] = bn; votes[slot] = 0.f; }
            if (slot >= 0) votes[slot] += w;
        }

        if (nPts == 0)
        {
            // no measurement around this audio: hold the note through short gaps (consonants), then pass through
            bool recent = false;
            const double hold = 0.06 * sr;
            for (long long k = trkN - 1; k >= 0 && k > trkN - 40; --k)
            {
                const TrackPt& p = trk[(size_t) (k % kTrk)];
                if (p.t <= tdes) { recent = (tdes - p.t) < hold; break; }
            }
            if (!recent || curNote < 0)
            {
                resVoiced = false; snapOn = false; curNote = -1;
                targetMidiOut = 69.f + 12.f * std::log2(((float) sr / T0cur) / 440.f);
                meterVoiced = false;
            }
            return;
        }

        // winner, with hysteresis towards the note we are already on
        int win = noteIds[0]; float wv = votes[0]; float curV = 0.f;
        for (int i = 0; i < nNotes; ++i)
        {
            if (votes[i] > wv) { wv = votes[i]; win = noteIds[i]; }
            if (noteIds[i] == curNote) curV = votes[i];
        }
        const float hyst = 1.f + 0.5f * trackAmt;
        if (curNote >= 0 && allowed(curNote) && win != curNote && wv < curV * hyst) win = curNote;
        curNote = win;
        const float best = (float) curNote;

        // pitch of the singer at tdes: median of the three nearest measurements (kills glitches)
        int idx[16]; for (int i = 0; i < nPts; ++i) idx[i] = i;
        std::sort(idx, idx + nPts, [&](int a, int b) { return std::abs(times[a] - tdes) < std::abs(times[b] - tdes); });
        float near3[3]; const int n3 = juce::jmin(3, nPts);
        for (int i = 0; i < n3; ++i) near3[i] = midis[idx[i]];
        std::sort(near3, near3 + n3);
        float m = near3[n3 / 2];
        float allM[16]; for (int i = 0; i < nPts; ++i) allM[i] = midis[i];
        std::sort(allM, allM + nPts);
        const float wmed = allM[nPts / 2];
        if (std::abs(m - wmed) > 1.f) m = wmed;

        float allT[16]; for (int i = 0; i < nPts; ++i) allT[i] = T0s[i];
        std::sort(allT, allT + nPts);
        T0cur = juce::jlimit((float) (sr / 800.0), (float) (sr / 90.0), allT[nPts / 2]);

        targetMidiOut = m + (best - m) * amountP;
        resM = m; resVoiced = true; snapOn = true;
        meterDetected = m; meterTarget = best; meterVoiced = true;
    }

    // PSOLA: grains of two periods, placed exactly at the spacing of the TARGET note
    void generateGrains(int nCh)
    {
        if (nextTs < (double) (wp - 4096)) nextTs = (double) wp;

        while (nextTs <= (double) wp + 1.15 * (double) T0cur)
        {
            const double tdes = nextTs - (double) delay0;
            resolveTarget(tdes);

            const double T = (double) T0cur;
            const int Ti = juce::jmax(8, (int) std::llround(T));

            int guard = 0;
            while (std::abs(anaMark + T - tdes) < std::abs(anaMark - tdes) && guard++ < 16)
                anaMark = nextAnaMark(anaMark, T);
            if (guard >= 16) anaMark = tdes;

            if (anaMark + (double) Ti + 2.0 > (double) wp) anaMark = (double) (wp - Ti - 2);
            if (anaMark - (double) Ti < (double) (wp - kBuf + 8)) anaMark = (double) (wp - kBuf + 8 + Ti);

            const double curFreq = 440.0 * std::pow(2.0, ((double) curMidiOut - 69.0) / 12.0);
            const double Ts = juce::jlimit(sr / 2000.0, sr / 40.0, sr / curFreq);
            outFreq = (float) curFreq;

            const long long tsInt = (long long) std::floor(nextTs + 0.5);
            const double frac = nextTs - (double) tsInt;

            for (int k = -Ti; k <= Ti; ++k)
            {
                const float w = 0.5f + 0.5f * std::cos(juce::MathConstants<float>::pi * (float) k / (float) Ti);
                const size_t idx = (size_t) ((tsInt + k) & kMask);
                wsum[idx] += w;
                for (int c = 0; c < nCh; ++c)
                    outB[(size_t) c][idx] += w * readAt(dl[(size_t) c], anaMark + (double) k - frac);
            }
            nextTs += Ts;
        }
    }

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout l;
        auto F = [&l](const char* id, const char* name, float lo, float hi, float step, float skew, float def)
        {
            l.add(std::make_unique<AudioParameterFloat>(ParameterID{id, 1}, name,
                  NormalisableRange<float>(lo, hi, step, skew), def));
        };

        // auto tune
        l.add(std::make_unique<AudioParameterBool>(ParameterID{"tune_on", 1}, "Auto Tune On", true));
        l.add(std::make_unique<AudioParameterChoice>(ParameterID{"key", 1}, "Key", keyNames(), 0));
        l.add(std::make_unique<AudioParameterChoice>(ParameterID{"scale", 1}, "Scale", scaleNames(), 2));
        F("retune", "Retune Speed", 0.f, 200.f, 0.1f, 0.5f, 0.f);
        F("track",  "Tracking",     0.f, 1.f, 0.001f, 1.f, 0.5f);
        F("amount", "Amount", 0.f, 1.f, 0.001f, 1.f, 1.f);
        F("robot",  "Robot",  0.f, 1.f, 0.001f, 1.f, 0.3f);
        F("grit",   "Grit",   0.f, 1.f, 0.001f, 1.f, 0.f);
        F("mix",    "Mix",    0.f, 1.f, 0.001f, 1.f, 1.f);
        F("output", "Output", -12.f, 12.f, 0.1f, 1.f, 0.f);

        // fresh air
        l.add(std::make_unique<AudioParameterBool>(ParameterID{"air_on", 1}, "Fresh Air On", false));
        F("air_amount",  "Air Amount",          0.f, 1.f, 0.001f, 1.f, 0.4f);
        F("air_freq",    "Air Shine Frequency", 3000.f, 12000.f, 1.f, 0.6f, 8000.f);
        F("air_sparkle", "Air Sparkle",         0.f, 1.f, 0.001f, 1.f, 0.3f);
        F("air_smooth",  "Air Smooth",          0.f, 1.f, 0.001f, 1.f, 0.5f);

        // equalizer
        l.add(std::make_unique<AudioParameterBool>(ParameterID{"eq_on", 1}, "EQ On", false));
        F("eq_lowcut",  "EQ Low Cut",       20.f, 400.f, 1.f, 0.4f, 20.f);
        F("eq_low",     "EQ Low Gain",      -12.f, 12.f, 0.1f, 1.f, 0.f);
        F("eq_low_f",   "EQ Low Freq",      40.f, 400.f, 1.f, 0.5f, 120.f);
        F("eq_lomid",   "EQ Low Mid Gain",  -12.f, 12.f, 0.1f, 1.f, 0.f);
        F("eq_lomid_f", "EQ Low Mid Freq",  150.f, 1500.f, 1.f, 0.5f, 350.f);
        F("eq_himid",   "EQ High Mid Gain", -12.f, 12.f, 0.1f, 1.f, 0.f);
        F("eq_himid_f", "EQ High Mid Freq", 1000.f, 8000.f, 1.f, 0.5f, 3500.f);
        F("eq_high",    "EQ High Gain",     -12.f, 12.f, 0.1f, 1.f, 0.f);
        F("eq_high_f",  "EQ High Freq",     3000.f, 16000.f, 1.f, 0.5f, 8000.f);
        F("eq_highcut", "EQ High Cut",      2000.f, 20000.f, 1.f, 0.4f, 20000.f);
        return l;
    }

    // Pitch measurement (YIN). It only records "at input time t the voice had pitch p";
    // the note decision is taken later, per grain, by resolveTarget().
    void detect()
    {
        const int N = (int) (0.0175 * sr);
        const int minTau = (int) (sr / 800.0);
        const int maxTau = (int) (sr / 90.0);
        const int total = N + maxTau;

        float energy = 0.f;
        for (int j = 0; j < total; ++j)
            scratch[(size_t) j] = det[(size_t) ((wp - total + j) & kMask)];
        for (int j = maxTau; j < total; ++j) energy += scratch[(size_t) j] * scratch[(size_t) j];

        bool voiced = false;
        float midiRaw = 0.f, conf = 0.f, tauEst = 0.f;

        if (std::sqrt(energy / (float) N) >= 0.0025f)
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
            // keep following a voice we were already tracking when it gets breathy or noisy
            if (found < 0 && lastDetVoiced)
            {
                const float tau0 = (float) sr / (440.f * std::pow(2.f, (lastDetMidi - 69.f) / 12.f));
                const int lo = juce::jmax(2, (int) (tau0 * 0.89f)), hi = juce::jmin(maxTau - 1, (int) (tau0 * 1.12f));
                float mn = 1e9f; int mt = -1;
                for (int tau = lo; tau <= hi; ++tau)
                    if (yin[(size_t) tau] < mn) { mn = yin[(size_t) tau]; mt = tau; }
                if (mt >= 2 && mn < 0.3f) found = mt;
            }

            // octave-down guard: if half of this period is also clearly periodic, the real period is the shorter one
            // (happens while the pitch is moving, when the true period fails the strict threshold)
            for (int pass = 0; pass < 2 && found >= 2; ++pass)
            {
                const int half = found / 2;
                if (half < juce::jmax(2, minTau)) break;
                const int lo2 = juce::jmax(2, (int) ((float) half * 0.93f)), hi2 = juce::jmin(maxTau - 1, (int) ((float) half * 1.07f) + 1);
                float mn = 1e9f; int mt = -1;
                for (int tau = lo2; tau <= hi2; ++tau)
                    if (yin[(size_t) tau] < mn) { mn = yin[(size_t) tau]; mt = tau; }
                if (mt >= 2 && mn < 0.4f && mn < yin[(size_t) found] * 3.f + 0.1f) found = mt; else break;
            }

            if (found >= 2)
            {
                tauEst = (float) found;
                const float s0 = yin[(size_t) found - 1], s1 = yin[(size_t) found], s2 = yin[(size_t) found + 1];
                const float den = s0 - 2.f * s1 + s2;
                if (std::abs(den) > 1e-9f) tauEst += 0.5f * (s0 - s2) / den;
                conf = juce::jlimit(0.05f, 1.f, 1.f - yin[(size_t) found]);
                midiRaw = 69.f + 12.f * std::log2(((float) sr / tauEst) / 440.f);
                voiced = true;
                // an octave-sized jump with low confidence is almost always a tracking error
                if (lastDetVoiced && std::abs(midiRaw - lastDetMidi) > 10.f && conf < 0.8f) voiced = false;
            }
        }

        lastDetVoiced = voiced;
        if (!voiced) return;
        lastDetMidi = midiRaw;

        TrackPt& p = trk[(size_t) (trkN % kTrk)];
        p.t = (double) wp - 0.5 * (double) N - 0.5 * (double) tauEst;   // centre of the analysed segment
        p.midi = midiRaw; p.conf = conf; p.T0 = tauEst;
        ++trkN;
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BlackTune)
};

//==============================================================================
static const juce::Colour kGold(0xfff5c542);
static const juce::Colour kRed(0xffe02020);
static const juce::Colour kGreen(0xff39d353);
static const juce::Colour kBand[4] = { juce::Colour(0xffff6b5a), juce::Colour(0xffffb347),
                                       juce::Colour(0xff6fd3a8), juce::Colour(0xff6fb7ff) };

// font with optional letter-spacing (tracking) for a refined look
static juce::Font F(float h, int style = juce::Font::plain, float kern = 0.f)
{
    return juce::Font(juce::FontOptions(h, style).withKerningFactor(kern));
}

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
                          float sliderPos, float startAngle, float endAngle, juce::Slider& slider) override
    {
        auto bounds = juce::Rectangle<float>((float) x, (float) y, (float) width, (float) height).reduced(10.f);
        const float r = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.f;
        const auto c = bounds.getCentre();
        const float ang = startAngle + sliderPos * (endAngle - startAngle);

        // knobs that go both ways (dB) draw their arc from the centre
        const bool bipolar = slider.getMinimum() < 0.0 && slider.getMaximum() > 0.0
                             && std::abs(slider.getMinimum() + slider.getMaximum()) < 1e-6;
        const float midAng = 0.5f * (startAngle + endAngle);
        const float arcFrom = bipolar ? juce::jmin(midAng, ang) : startAngle;
        const float arcTo   = bipolar ? juce::jmax(midAng, ang) : ang;

        for (int i = 0; i <= 20; ++i)   // scale ticks
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
        g.setColour(juce::Colour(0xff1d1d1d));
        g.strokePath(track, juce::PathStrokeType(5.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        g.setColour(juce::Colour(0xff2e2e2e));
        g.strokePath(track, juce::PathStrokeType(3.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        if (std::abs(arcTo - arcFrom) > 0.01f)
        {
            juce::Path val;
            val.addCentredArc(c.x, c.y, arcR, arcR, 0.f, arcFrom, arcTo, true);
            g.setColour(kGold.withAlpha(0.14f));
            g.strokePath(val, juce::PathStrokeType(12.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            g.setColour(kGold.withAlpha(0.28f));
            g.strokePath(val, juce::PathStrokeType(7.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            g.setColour(juce::Colour(0xffffd45c));
            g.strokePath(val, juce::PathStrokeType(3.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }

        const float bezelR = arcR - 5.f;
        for (int i = 4; i >= 1; --i)   // soft drop shadow
        {
            g.setColour(juce::Colours::black.withAlpha(0.13f));
            g.fillEllipse(c.x - bezelR - (float) i, c.y - bezelR - (float) i + 3.f,
                          2.f * bezelR + 2.f * (float) i, 2.f * bezelR + 2.f * (float) i);
        }

        // chrome bezel
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xffd2d2d2), c.x - bezelR * 0.6f, c.y - bezelR * 0.8f,
                                               juce::Colour(0xff1c1c1c), c.x + bezelR * 0.7f, c.y + bezelR * 0.9f, false));
        g.fillEllipse(c.x - bezelR, c.y - bezelR, bezelR * 2.f, bezelR * 2.f);
        g.setColour(juce::Colours::black.withAlpha(0.65f));
        g.drawEllipse(c.x - bezelR, c.y - bezelR, bezelR * 2.f, bezelR * 2.f, 1.f);

        g.setColour(juce::Colours::black.withAlpha(0.5f));   // knurled edge
        for (int i = 0; i < 48; ++i)
        {
            const float a = (float) i * juce::MathConstants<float>::twoPi / 48.f;
            const float s = std::sin(a), co = std::cos(a);
            g.drawLine(c.x + s * (bezelR - 2.2f), c.y - co * (bezelR - 2.2f),
                       c.x + s * (bezelR - 0.4f), c.y - co * (bezelR - 0.4f), 1.f);
        }

        // brushed cap
        const float capR = bezelR - 4.f;
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff5c5c5c), c.x - capR * 0.4f, c.y - capR * 0.5f,
                                               juce::Colour(0xff0f0f0f), c.x + capR * 0.8f, c.y + capR * 0.95f, true));
        g.fillEllipse(c.x - capR, c.y - capR, capR * 2.f, capR * 2.f);
        for (float rr = capR * 0.3f; rr < capR; rr += 2.2f)
        {
            g.setColour(juce::Colours::white.withAlpha(0.025f));
            g.drawEllipse(c.x - rr, c.y - rr, rr * 2.f, rr * 2.f, 0.7f);
        }
        g.setGradientFill(juce::ColourGradient(juce::Colours::white.withAlpha(0.20f), c.x, c.y - capR,
                                               juce::Colours::transparentWhite, c.x, c.y - capR * 0.1f, false));
        g.fillEllipse(c.x - capR * 0.75f, c.y - capR * 0.95f, capR * 1.5f, capR * 0.9f);

        // pointer with glow
        juce::Path p;
        p.addRoundedRectangle(-1.7f, -capR + 2.f, 3.4f, capR * 0.5f, 1.5f);
        const auto xf = juce::AffineTransform::rotation(ang).translated(c.x, c.y);
        g.setColour(kGold.withAlpha(0.3f));
        g.strokePath(p, juce::PathStrokeType(3.f), xf);
        g.setColour(juce::Colour(0xffffd45c));
        g.fillPath(p, xf);
    }

    void drawButtonBackground(juce::Graphics& g, juce::Button& b, const juce::Colour&,
                              bool highlighted, bool down) override
    {
        auto r = b.getLocalBounds().toFloat().reduced(0.5f);

        if (b.getProperties().contains("tab"))
        {
            const bool sel = b.getToggleState();
            juce::Colour top = sel ? juce::Colour(0xff303030) : juce::Colour(0xff1a1a1a);
            const juce::Colour bot = sel ? juce::Colour(0xff121212) : juce::Colour(0xff0a0a0a);
            if (highlighted && !sel) top = top.brighter(0.2f);
            g.setGradientFill(juce::ColourGradient(top, 0.f, r.getY(), bot, 0.f, r.getBottom(), false));
            g.fillRoundedRectangle(r, 8.f);
            g.setColour(juce::Colours::white.withAlpha(sel ? 0.08f : 0.03f));
            g.fillRoundedRectangle(r.reduced(1.f).withHeight(r.getHeight() * 0.48f), 7.f);
            g.setColour(kGold.withAlpha(sel ? 0.8f : 0.25f));
            g.drawRoundedRectangle(r, 8.f, 1.f);
            if (sel)
            {
                g.setColour(kGold);
                g.fillRoundedRectangle(r.getX() + 14.f, r.getBottom() - 4.f, r.getWidth() - 28.f, 3.f, 1.5f);
            }
            return;
        }

        juce::Colour top(0xff2d2d2d), bot(0xff0d0d0d);
        if (highlighted) top = top.brighter(0.25f);
        if (down) std::swap(top, bot);
        g.setGradientFill(juce::ColourGradient(top, 0.f, r.getY(), bot, 0.f, r.getBottom(), false));
        g.fillRoundedRectangle(r, 5.f);
        g.setColour(kGold.withAlpha(highlighted ? 0.95f : 0.45f));
        g.drawRoundedRectangle(r, 5.f, 1.f);
    }

    juce::Font getTextButtonFont(juce::TextButton& b, int) override
    {
        return F(b.getProperties().contains("tab") ? 14.f : 11.f, juce::Font::bold, 0.08f);
    }

    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& b, bool highlighted, bool) override
    {
        auto r = b.getLocalBounds().toFloat().reduced(0.5f);
        const bool on = b.getToggleState();
        juce::Colour top = on ? juce::Colour(0xff2a3a2a) : juce::Colour(0xff2d2d2d);
        const juce::Colour bot(0xff0d0d0d);
        if (highlighted) top = top.brighter(0.2f);
        g.setGradientFill(juce::ColourGradient(top, 0.f, r.getY(), bot, 0.f, r.getBottom(), false));
        g.fillRoundedRectangle(r, 5.f);
        g.setColour((on ? kGreen : kGold).withAlpha(highlighted ? 0.95f : 0.55f));
        g.drawRoundedRectangle(r, 5.f, 1.f);

        auto led = juce::Rectangle<float>(r.getX() + 9.f, r.getCentreY() - 5.f, 10.f, 10.f);
        if (on)
        {
            g.setColour(kGreen.withAlpha(0.3f));
            g.fillEllipse(led.expanded(4.f));
        }
        g.setColour(on ? kGreen : juce::Colour(0xff3a3a3a));
        g.fillEllipse(led);
        g.setColour(juce::Colours::black.withAlpha(0.7f));
        g.drawEllipse(led, 1.f);

        g.setColour(on ? juce::Colours::white : juce::Colour(0xff9a9a9a));
        g.setFont(F(12.f, juce::Font::bold));
        g.drawText(on ? "ON" : "OFF", r.withTrimmedLeft(24.f).toNearestInt(), juce::Justification::centred);
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
        setSize(800, 592);
        setLookAndFeel(&laf);

        // ---- tabs ----
        const char* tn[3] = { "AUTO TUNE", "FRESH AIR", "EQUALIZER" };
        for (int i = 0; i < 3; ++i)
        {
            auto& t = tabs[(size_t) i];
            t.setButtonText(tn[i]);
            t.getProperties().set("tab", true);
            t.setClickingTogglesState(true);
            t.setRadioGroupId(101);
            t.setColour(juce::TextButton::textColourOffId, juce::Colour(0xff9a9a9a));
            t.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
            t.onClick = [this, i] { setPage(i); };
            addAndMakeVisible(t);
        }
        tabs[0].setToggleState(true, juce::dontSendNotification);

        // ---- auto tune page ----
        const char* tIds[7]   = { "retune", "track", "amount", "robot", "grit", "mix", "output" };
        const char* tNames[7] = { "RETUNE SPEED", "TRACKING", "AMOUNT", "ROBOT", "GRIT", "MIX", "OUTPUT" };
        const int tKinds[7]   = { 1, 0, 0, 0, 0, 0, 2 };
        for (int i = 0; i < 7; ++i) setupKnob(knobs[(size_t) i], tIds[i], tNames[i], tKinds[i], i == 0);

        setupCombo(keyBox, keyNames());
        setupCombo(scaleBox, scaleNames());
        keyA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, "key", keyBox);
        scaleA = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(proc.apvts, "scale", scaleBox);

        const char* pn[5] = { "HARD TUNE", "ROBOT", "CLEAN", "NATURAL", "DESTROY" };
        for (int i = 0; i < 5; ++i) setupPreset(presets[(size_t) i], pn[i], [this, i] { applyTunePreset(i); });

        // ---- fresh air page ----
        const char* aNames[4] = { "AIR", "SHINE", "SPARKLE", "SMOOTH" };
        const int aKinds[4]   = { 0, 3, 0, 0 };
        for (int i = 0; i < 4; ++i) setupKnob(airK[(size_t) i], kAirIds[i], aNames[i], aKinds[i], false);
        const char* apn[3] = { "NATURAL", "BRIGHT", "SILKY" };
        for (int i = 0; i < 3; ++i) setupPreset(airPresets[(size_t) i], apn[i], [this, i] { applyAirPreset(i); });

        // ---- equalizer page ----
        const char* eNames[10] = { "LOW CUT", "GAIN", "FREQ", "GAIN", "FREQ", "GAIN", "FREQ", "GAIN", "FREQ", "HIGH CUT" };
        const int eKinds[10]   = { 5, 2, 4, 2, 4, 2, 4, 2, 4, 6 };
        for (int i = 0; i < 10; ++i) setupKnob(eqK[(size_t) i], kEqIds[i], eNames[i], eKinds[i], false);
        const char* epn[4] = { "FLAT", "VOCAL", "WARM", "RADIO" };
        for (int i = 0; i < 4; ++i) setupPreset(eqPresets[(size_t) i], epn[i], [this, i] { applyEqPreset(i); });

        // ---- power buttons ----
        addAndMakeVisible(tuneBtn);
        addAndMakeVisible(airBtn);
        addAndMakeVisible(eqBtn);
        tuneA = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(proc.apvts, "tune_on", tuneBtn);
        airA  = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(proc.apvts, "air_on", airBtn);
        eqA   = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(proc.apvts, "eq_on", eqBtn);

        // ---- which components belong to which page ----
        auto reg = [this](int pg, juce::Component& c) { pageComps[pg].push_back(&c); };
        for (auto& k : knobs) { reg(0, k.s); reg(0, k.l); }
        reg(0, keyBox); reg(0, scaleBox); reg(0, tuneBtn);
        for (auto& b : presets) reg(0, b);
        for (auto& k : airK) { reg(1, k.s); reg(1, k.l); }
        reg(1, airBtn);
        for (auto& b : airPresets) reg(1, b);
        for (auto& k : eqK) { reg(2, k.s); reg(2, k.l); }
        reg(2, eqBtn);
        for (auto& b : eqPresets) reg(2, b);

        setPage(0);
        startTimerHz(30);
    }

    ~BlackTuneEditor() override { setLookAndFeel(nullptr); }

    void resized() override
    {
        for (int i = 0; i < 3; ++i) tabs[(size_t) i].setBounds(20 + i * 254, 80, 252, 34);

        // auto tune page
        for (int i = 0; i < 7; ++i)
        {
            juce::Rectangle<int> col(22 + i * 108, 436, 108, 134);
            knobs[(size_t) i].l.setBounds(col.removeFromTop(20));
            knobs[(size_t) i].s.setBounds(col);
        }
        keyBox.setBounds(58, 390, 78, 28);
        scaleBox.setBounds(198, 390, 150, 28);
        for (int i = 0; i < 5; ++i) presets[(size_t) i].setBounds(362 + i * 66, 390, 62, 28);
        tuneBtn.setBounds(700, 390, 70, 28);

        // fresh air page
        airBtn.setBounds(700, 134, 70, 26);
        for (int i = 0; i < 3; ++i) airPresets[(size_t) i].setBounds(412 + i * 84, 134, 80, 26);
        for (int i = 0; i < 4; ++i)
        {
            juce::Rectangle<int> col(36 + i * 182, 376, 182, 184);
            airK[(size_t) i].l.setBounds(col.removeFromTop(20));
            airK[(size_t) i].s.setBounds(col);
        }

        // equalizer page
        eqBtn.setBounds(700, 134, 70, 26);
        for (int i = 0; i < 4; ++i) eqPresets[(size_t) i].setBounds(412 + i * 66, 134, 62, 26);
        for (int i = 0; i < 10; ++i)
        {
            juce::Rectangle<int> col(36 + i * 73, 358, 73, 202);
            eqK[(size_t) i].l.setBounds(col.removeFromTop(20));
            eqK[(size_t) i].s.setBounds(col);
        }
    }

    void paint(juce::Graphics& g) override
    {
        const int W = getWidth(), H = getHeight();

        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff202020), 0.f, 0.f,
                                               juce::Colour(0xff060606), 0.f, (float) H, false));
        g.fillAll();
        g.setColour(juce::Colours::white.withAlpha(0.018f));
        for (int y = 0; y < H; y += 3) g.drawHorizontalLine(y, 0.f, (float) W);
        g.setGradientFill(juce::ColourGradient(juce::Colours::transparentBlack, (float) W * 0.5f, (float) H * 0.5f,
                                               juce::Colours::black.withAlpha(0.55f), 0.f, 0.f, true));
        g.fillAll();

        drawHeader(g, W);

        if (page == 0)      drawTunePage(g);
        else if (page == 1) drawAirPage(g);
        else                drawEqPage(g);
    }

    // power LEDs on the tabs are drawn above the tab buttons
    void paintOverChildren(juce::Graphics& g) override
    {
        const char* ids[3] = { "tune_on", "air_on", "eq_on" };
        for (int i = 0; i < 3; ++i)
        {
            const bool on = proc.apvts.getRawParameterValue(ids[i])->load() > 0.5f;
            const auto b = tabs[(size_t) i].getBounds().toFloat();
            const auto led = juce::Rectangle<float>(b.getX() + 18.f, b.getCentreY() - 5.f, 10.f, 10.f);
            if (on)
            {
                g.setColour(kGreen.withAlpha(0.3f));
                g.fillEllipse(led.expanded(4.f));
            }
            g.setColour(on ? kGreen : juce::Colour(0xff3a3a3a));
            g.fillEllipse(led);
            g.setColour(juce::Colours::black.withAlpha(0.7f));
            g.drawEllipse(led, 1.f);
        }
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
    std::array<Knob, 7> knobs;
    std::array<Knob, 4> airK;
    std::array<Knob, 10> eqK;
    juce::ComboBox keyBox, scaleBox;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> keyA, scaleA;
    std::array<juce::TextButton, 5> presets;
    std::array<juce::TextButton, 3> airPresets;
    std::array<juce::TextButton, 4> eqPresets;
    std::array<juce::TextButton, 3> tabs;
    juce::ToggleButton tuneBtn, airBtn, eqBtn;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> tuneA, airA, eqA;
    std::vector<juce::Component*> pageComps[3];
    int page = 0;
    float viewC = 62.f;   // centre note of the pitch graph
    int lastOn[3] = { -1, -1, -1 };

    // tune page
    const juce::Rectangle<int> meterPanel { 20, 124, 760, 94 };
    const juce::Rectangle<int> graphPanel { 20, 226, 760, 150 };
    const juce::Rectangle<int> stripPanel { 20, 384, 760, 40 };
    const juce::Rectangle<int> knobPanel { 20, 432, 760, 144 };
    // air + eq pages
    const juce::Rectangle<int> fxPanel { 20, 124, 760, 444 };
    const juce::Rectangle<int> airCurve { 36, 172, 728, 170 };
    const juce::Rectangle<int> airBar { 36, 350, 728, 18 };
    const juce::Rectangle<int> eqCurve { 36, 172, 728, 160 };

    //==========================================================================
    void setPage(int idx)
    {
        page = idx;
        for (int p = 0; p < 3; ++p)
            for (auto* c : pageComps[p]) c->setVisible(p == idx);
        repaint();
    }

    void timerCallback() override
    {
        if (page == 0)
        {
            // slowly re-centre the pitch graph on the notes being sung
            float sum = 0.f; int cnt = 0;
            const int pos = proc.trailPos.load();
            for (int i = 0; i < 48; ++i)
            {
                const float v = proc.trailOut[(size_t) ((pos - 1 - i + BlackTune::kTrail * 2) % BlackTune::kTrail)].load();
                if (v > 0.f) { sum += v; ++cnt; }
            }
            if (cnt > 4) viewC += 0.08f * (sum / (float) cnt - viewC);
            repaint(meterPanel.getUnion(graphPanel));
        }
        else if (page == 1) repaint(airCurve.getUnion(airBar));
        else                repaint(eqCurve);

        const char* ids[3] = { "tune_on", "air_on", "eq_on" };
        for (int i = 0; i < 3; ++i)
        {
            const int on = proc.apvts.getRawParameterValue(ids[i])->load() > 0.5f ? 1 : 0;
            if (on != lastOn[i])
            {
                lastOn[i] = on;
                const float a = on ? 1.f : 0.4f;
                if (i == 0) for (auto& k : knobs) { k.s.setAlpha(a); k.l.setAlpha(a); }
                if (i == 1) for (auto& k : airK)  { k.s.setAlpha(a); k.l.setAlpha(a); }
                if (i == 2) for (auto& k : eqK)   { k.s.setAlpha(a); k.l.setAlpha(a); }
                repaint();
            }
        }
    }

    void setupCombo(juce::ComboBox& box, const juce::StringArray& items)
    {
        box.addItemList(items, 1);
        box.setColour(juce::ComboBox::textColourId, juce::Colours::white);
        addAndMakeVisible(box);
    }

    void setupPreset(juce::TextButton& b, const char* name, std::function<void()> fn)
    {
        b.setButtonText(name);
        b.setColour(juce::TextButton::textColourOffId, kGold);
        b.setColour(juce::TextButton::textColourOnId, kGold);
        b.onClick = std::move(fn);
        addAndMakeVisible(b);
    }

    // kind: 0 percent, 1 ms, 2 dB, 3 kHz, 4 Hz/kHz, 5 low cut (OFF at minimum), 6 high cut (OFF at maximum)
    void setupKnob(Knob& k, const char* id, const char* name, int kind, bool gold)
    {
        k.s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        k.s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 84, 18);
        k.s.setRotaryParameters(juce::MathConstants<float>::pi * 1.2f,
                                juce::MathConstants<float>::pi * 2.8f, true);
        k.s.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe8e8e8));
        k.s.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        addAndMakeVisible(k.s);

        k.l.setText(name, juce::dontSendNotification);
        k.l.setJustificationType(juce::Justification::centred);
        k.l.setColour(juce::Label::textColourId, gold ? kGold : juce::Colour(0xffbdbdbd));
        k.l.setFont(F(12.f, juce::Font::bold, 0.08f));
        addAndMakeVisible(k.l);

        k.a = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, id, k.s);

        auto hz = [](double v) -> juce::String
        {
            return v >= 1000.0 ? juce::String(v / 1000.0, 2) + " kHz" : juce::String((int) std::round(v)) + " Hz";
        };

        if (kind == 0)
        {
            k.s.textFromValueFunction = [](double v) { return juce::String((int) std::round(v * 100.0)) + " %"; };
            k.s.valueFromTextFunction = [](const juce::String& t) { return t.getDoubleValue() / 100.0; };
        }
        else
        {
            if (kind == 1)      k.s.textFromValueFunction = [](double v) { return juce::String(v, 1) + " ms"; };
            else if (kind == 2) k.s.textFromValueFunction = [](double v) { return juce::String(v, 1) + " dB"; };
            else if (kind == 3) k.s.textFromValueFunction = [](double v) { return juce::String(v / 1000.0, 1) + " kHz"; };
            else if (kind == 4) k.s.textFromValueFunction = [hz](double v) { return hz(v); };
            else if (kind == 5) k.s.textFromValueFunction = [hz](double v) { return v <= 20.5 ? juce::String("OFF") : hz(v); };
            else                k.s.textFromValueFunction = [hz](double v) { return v >= 19900.0 ? juce::String("OFF") : hz(v); };

            k.s.valueFromTextFunction = [kind](const juce::String& t)
            {
                const double v = t.getDoubleValue();
                if (kind == 3) return v * 1000.0;
                if (kind >= 4 && t.containsIgnoreCase("k")) return v * 1000.0;
                return v;
            };
        }
        k.s.updateText();
    }

    void setParam(const char* id, float v)
    {
        if (auto* prm = proc.apvts.getParameter(id))
        {
            prm->beginChangeGesture();
            prm->setValueNotifyingHost(prm->convertTo0to1(v));
            prm->endChangeGesture();
        }
    }

    void applyTunePreset(int idx)
    {
        static const float P[5][5] = {
            { 0.f,  1.f,  0.25f, 0.f,   1.f },   // HARD TUNE
            { 0.f,  1.f,  0.75f, 0.15f, 1.f },   // ROBOT
            { 15.f, 1.f,  0.f,   0.f,   1.f },   // CLEAN
            { 60.f, 0.8f, 0.f,   0.f,   1.f },   // NATURAL
            { 0.f,  1.f,  0.9f,  0.6f,  1.f }    // DESTROY
        };
        const char* ids[5] = { "retune", "amount", "robot", "grit", "mix" };
        for (int i = 0; i < 5; ++i) setParam(ids[i], P[idx][i]);
        setParam("tune_on", 1.f);
    }

    void applyAirPreset(int idx)
    {
        static const float P[3][4] = {
            { 0.30f, 9000.f, 0.20f, 0.60f },   // NATURAL
            { 0.55f, 8000.f, 0.45f, 0.50f },   // BRIGHT
            { 0.45f, 6500.f, 0.15f, 0.80f }    // SILKY
        };
        for (int i = 0; i < 4; ++i) setParam(kAirIds[i], P[idx][i]);
        setParam("air_on", 1.f);
    }

    void applyEqPreset(int idx)
    {
        //                 lowcut low  lowF  lomid lomidF himid himidF high  highF highcut
        static const float P[4][10] = {
            { 20.f,  0.f,   120.f, 0.f,   350.f, 0.f,  3500.f, 0.f,  8000.f,  20000.f },  // FLAT
            { 90.f, -1.f,   120.f, -2.5f, 320.f, 2.f,  3500.f, 3.f,  10000.f, 20000.f },  // VOCAL
            { 70.f,  2.5f,  150.f, 1.f,   400.f, -1.f, 3000.f, 1.5f, 9000.f,  18000.f },  // WARM
            { 300.f, 0.f,   120.f, 3.f,   900.f, 4.f,  3000.f, -6.f, 8000.f,  9000.f }    // RADIO
        };
        for (int i = 0; i < 10; ++i) setParam(kEqIds[i], P[idx][i]);
        setParam("eq_on", 1.f);
    }

    //==========================================================================
    static juce::String noteName(int midiNote)
    {
        return keyNames()[((midiNote % 12) + 12) % 12];
    }

    void drawScrew(juce::Graphics& g, float cx, float cy)
    {
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff9a9a9a), cx - 2.f, cy - 2.f,
                                               juce::Colour(0xff222222), cx + 3.f, cy + 3.f, false));
        g.fillEllipse(cx - 3.5f, cy - 3.5f, 7.f, 7.f);
        g.setColour(juce::Colours::black.withAlpha(0.7f));
        g.drawEllipse(cx - 3.5f, cy - 3.5f, 7.f, 7.f, 0.8f);
        g.drawLine(cx - 2.f, cy + 1.2f, cx + 2.f, cy - 1.2f, 1.1f);
    }

    void drawPanel(juce::Graphics& g, juce::Rectangle<int> r, bool screws = false)
    {
        auto f = r.toFloat();
        for (int i = 5; i >= 1; --i)   // soft drop shadow
        {
            g.setColour(juce::Colours::black.withAlpha(0.09f));
            g.fillRoundedRectangle(f.expanded((float) i).translated(0.f, (float) i * 0.7f), 10.f + (float) i);
        }
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff252525), f.getX(), f.getY(),
                                               juce::Colour(0xff090909), f.getX(), f.getBottom(), false));
        g.fillRoundedRectangle(f, 10.f);

        g.saveState();   // fine brushed-metal lines
        g.reduceClipRegion(r);
        g.setColour(juce::Colours::white.withAlpha(0.012f));
        for (int y = r.getY(); y < r.getBottom(); y += 2) g.drawHorizontalLine(y, f.getX(), f.getRight());
        g.restoreState();

        g.setColour(juce::Colours::white.withAlpha(0.05f));
        g.drawRoundedRectangle(f.reduced(1.5f), 9.f, 1.f);
        g.setGradientFill(juce::ColourGradient(kGold.withAlpha(0.6f), 0.f, f.getY(),
                                               kGold.withAlpha(0.14f), 0.f, f.getBottom(), false));
        g.drawRoundedRectangle(f.reduced(0.5f), 10.f, 1.2f);
        g.setGradientFill(juce::ColourGradient(juce::Colours::white.withAlpha(0.05f), 0.f, f.getY(),
                                               juce::Colours::transparentWhite, 0.f, f.getY() + f.getHeight() * 0.35f, false));
        g.fillRoundedRectangle(f.reduced(1.f).withHeight(f.getHeight() * 0.35f), 9.f);

        if (screws)
        {
            drawScrew(g, f.getX() + 9.f, f.getY() + 9.f);
            drawScrew(g, f.getRight() - 9.f, f.getY() + 9.f);
            drawScrew(g, f.getX() + 9.f, f.getBottom() - 9.f);
            drawScrew(g, f.getRight() - 9.f, f.getBottom() - 9.f);
        }
    }

    void drawHeader(juce::Graphics& g, int W)
    {
        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff141414), 0.f, 0.f,
                                               juce::Colour(0xff050505), 0.f, 68.f, false));
        g.fillRect(0, 0, W, 68);
        g.setGradientFill(juce::ColourGradient(juce::Colours::white.withAlpha(0.06f), 0.f, 0.f,
                                               juce::Colours::transparentWhite, 0.f, 34.f, false));
        g.fillRect(0, 0, W, 34);

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
        g.setFont(F(32.f, juce::Font::bold, 0.03f));
        g.drawText("BLACK TUNE RECORD", 84, 0, 480, 68, juce::Justification::centredLeft);

        g.setColour(juce::Colour(0xff9a9a9a));
        g.setFont(F(12.f, juce::Font::bold));
        g.drawText("VOCAL PROCESSOR", W - 320, 0, 300, 68, juce::Justification::centredRight);
    }

    //==========================================================================
    void drawTunePage(juce::Graphics& g)
    {
        drawPanel(g, meterPanel);
        drawPanel(g, graphPanel, true);
        drawPanel(g, stripPanel);
        drawPanel(g, knobPanel, true);
        drawMeter(g);
        drawPitchGraph(g);

        g.setColour(juce::Colour(0xffbdbdbd));
        g.setFont(F(12.f, juce::Font::bold, 0.1f));
        g.drawText("KEY", 30, 390, 28, 28, juce::Justification::centredLeft);
        g.drawText("SCALE", 144, 390, 52, 28, juce::Justification::centredLeft);
    }

    void drawMeter(juce::Graphics& g)
    {
        auto r = meterPanel.toFloat();
        const bool tuneOn = proc.apvts.getRawParameterValue("tune_on")->load() > 0.5f;
        const bool voiced = proc.meterVoiced.load();
        const float det = proc.meterDetected.load();
        const float tgt = proc.meterTarget.load();
        const float devCents = voiced ? juce::jlimit(-50.f, 50.f, (det - tgt) * 100.f) : 0.f;

        g.setColour(juce::Colour(0xff777777));
        g.setFont(F(10.f, juce::Font::bold, 0.12f));
        g.drawText("TARGET NOTE", (int) r.getX() + 18, (int) r.getY() + 8, 160, 14, juce::Justification::left);

        const juce::String nt = voiced ? noteName((int) std::round(tgt)) : "--";
        g.setFont(F(56.f, juce::Font::bold));
        const juce::Rectangle<int> nr((int) r.getX() + 10, (int) r.getY() + 16, 180, 58);
        if (voiced && tuneOn)
        {
            g.setColour(kGold.withAlpha(0.14f));
            for (int dx = -3; dx <= 3; dx += 3)
                for (int dy = -3; dy <= 3; dy += 3)
                    g.drawText(nt, nr.translated(dx, dy), juce::Justification::centred);
        }
        g.setColour(voiced && tuneOn ? kGold : juce::Colour(0xff4a4a4a));
        g.drawText(nt, nr, juce::Justification::centred);

        juce::String status = "NO SIGNAL";
        juce::Colour statusCol(0xff555555);
        if (!tuneOn)
            status = "BYPASSED";
        else if (voiced)
        {
            if (std::abs(devCents) <= 8.f) { status = "LOCKED"; statusCol = kGreen; }
            else if (devCents > 0.f)       { status = "SHARP";  statusCol = kRed; }
            else                           { status = "FLAT";   statusCol = kRed; }
        }
        g.setColour(statusCol);
        g.setFont(F(11.f, juce::Font::bold, 0.12f));
        g.drawText(status, (int) r.getX() + 10, (int) r.getY() + 74, 180, 16, juce::Justification::centred);

        // LED tuner bar
        const float bx = r.getX() + 215.f, bw = 450.f, by = r.getY() + 34.f, bh = 26.f;
        const int segs = 41;
        const float segW = bw / (float) segs;
        const int centerIdx = segs / 2;
        const int devIdx = centerIdx + (int) std::round(devCents / 2.5f);
        for (int i = 0; i < segs; ++i)
        {
            const float cents = (float) (i - centerIdx) * 2.5f;
            juce::Colour on = std::abs(cents) <= 10.f ? kGreen : std::abs(cents) <= 25.f ? kGold : kRed;
            bool lit = false;
            if (voiced && tuneOn)
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
                g.setColour(i == centerIdx ? juce::Colour(0xff3a3a3a) : juce::Colour(0xff161616));
            g.fillRoundedRectangle(seg, 2.f);
            g.setColour(juce::Colours::white.withAlpha(lit ? 0.18f : 0.04f));   // glass highlight
            g.fillRoundedRectangle(seg.withHeight(seg.getHeight() * 0.42f), 2.f);
        }
        g.setColour(juce::Colour(0xff777777));
        g.setFont(F(10.f, juce::Font::bold));
        g.drawText("-50", (int) bx - 10, (int) (by + bh + 3.f), 40, 12, juce::Justification::left);
        g.drawText("0", (int) (bx + bw * 0.5f) - 20, (int) (by + bh + 3.f), 40, 12, juce::Justification::centred);
        g.drawText("+50", (int) (bx + bw) - 30, (int) (by + bh + 3.f), 40, 12, juce::Justification::right);
        g.setFont(F(10.f, juce::Font::bold, 0.12f));
        g.drawText("CENTS", (int) bx, (int) r.getY() + 8, 100, 14, juce::Justification::left);

        if (voiced)
        {
            g.setColour(juce::Colours::white);
            g.setFont(F(13.f, juce::Font::bold));
            g.drawText(juce::String((int) std::round((det - tgt) * 100.f)) + " cents",
                       (int) bx, (int) r.getY() + 8, (int) bw, 16, juce::Justification::centred);
        }

        // output level meter
        const float lvx = r.getRight() - 46.f;
        const float db = juce::Decibels::gainToDecibels(proc.outLevel.load(), -60.f);
        const int lit = (int) std::round(juce::jlimit(0.f, 12.f, (db + 60.f) / 60.f * 12.f));
        for (int i = 0; i < 12; ++i)
        {
            juce::Colour on = i < 8 ? kGreen : i < 10 ? kGold : kRed;
            auto seg = juce::Rectangle<float>(lvx, r.getY() + 8.f + (float) (11 - i) * 5.6f, 20.f, 4.2f);
            g.setColour(i < lit ? on : juce::Colour(0xff161616));
            g.fillRoundedRectangle(seg, 1.5f);
        }
        g.setColour(juce::Colour(0xff777777));
        g.setFont(F(10.f, juce::Font::bold, 0.12f));
        g.drawText("OUT", (int) lvx - 10, (int) r.getBottom() - 20, 40, 12, juce::Justification::centred);
    }

    static bool isBlackKey(int n)
    {
        const int k = ((n % 12) + 12) % 12;
        return k == 1 || k == 3 || k == 6 || k == 8 || k == 10;
    }

    // piano-roll pitch graph: what you sing (white) and what the plugin outputs (gold)
    void drawPitchGraph(juce::Graphics& g)
    {
        const auto panel = graphPanel.toFloat();
        const float keysW = 54.f;
        const auto plot = juce::Rectangle<float>(panel.getX() + 12.f + keysW, panel.getY() + 10.f,
                                                 panel.getWidth() - 24.f - keysW, panel.getHeight() - 20.f);
        const auto keys = juce::Rectangle<float>(panel.getX() + 12.f, plot.getY(), keysW, plot.getHeight());

        const bool tuneOn = proc.apvts.getRawParameterValue("tune_on")->load() > 0.5f;
        const int mask = scaleMask((int) proc.apvts.getRawParameterValue("scale")->load());
        const int key = (int) proc.apvts.getRawParameterValue("key")->load();
        auto allowed = [&](int note) { return ((mask >> (((note - key) % 12 + 12) % 12)) & 1) != 0; };

        const float span = 18.f;
        const float lo = viewC - span * 0.5f;
        auto yOf = [&](float m) { return plot.getBottom() - plot.getHeight() * (m - lo) / span; };
        const bool voiced = proc.meterVoiced.load();
        const int tgtNote = voiced ? (int) std::round(proc.meterTarget.load()) : -999;

        g.saveState();
        g.reduceClipRegion(juce::Rectangle<float>(keys.getX(), plot.getY(), keys.getWidth() + 4.f + plot.getWidth(),
                                                  plot.getHeight()).toNearestInt());
        g.setColour(juce::Colour(0xff070707));
        g.fillRect(plot);

        for (int n = (int) std::floor(lo) - 1; n <= (int) std::ceil(lo + span) + 1; ++n)
        {
            const float yT = yOf((float) n + 0.5f), yB = yOf((float) n - 0.5f);
            const bool al = allowed(n), blk = isBlackKey(n), tg = (n == tgtNote) && tuneOn;

            // plot row
            const auto row = juce::Rectangle<float>(plot.getX(), yT, plot.getWidth(), yB - yT);
            g.setColour(tg ? kGold.withAlpha(0.16f)
                        : al ? juce::Colours::white.withAlpha(blk ? 0.035f : 0.06f)
                             : juce::Colours::black.withAlpha(0.35f));
            g.fillRect(row);
            g.setColour(juce::Colours::white.withAlpha(((n % 12) + 12) % 12 == 0 ? 0.16f : 0.05f));
            g.drawHorizontalLine((int) yB, plot.getX(), plot.getRight());

            // piano key
            const auto kr = juce::Rectangle<float>(keys.getX(), yT, keys.getWidth(), yB - yT);
            g.setGradientFill(juce::ColourGradient(al ? juce::Colour(0xfff0f0f0) : juce::Colour(0xff555555), kr.getX(), 0.f,
                                                   al ? juce::Colour(0xffb8b8b8) : juce::Colour(0xff2b2b2b), kr.getRight(), 0.f, false));
            g.fillRect(kr);
            if (blk)
            {
                const auto bk = juce::Rectangle<float>(kr.getX(), yT + 0.5f, kr.getWidth() * 0.64f, kr.getHeight() - 1.f);
                g.setGradientFill(juce::ColourGradient(al ? juce::Colour(0xff9a1c1c) : juce::Colour(0xff1c1c1c), bk.getX(), 0.f,
                                                       al ? juce::Colour(0xff3d0808) : juce::Colour(0xff050505), bk.getRight(), 0.f, false));
                g.fillRect(bk);
            }
            if (tg)
            {
                g.setColour(kGold.withAlpha(0.65f));
                g.fillRect(kr);
            }
            g.setColour(juce::Colours::black.withAlpha(0.7f));
            g.drawHorizontalLine((int) yB, kr.getX(), kr.getRight());
            if (((n % 12) + 12) % 12 == 0)
            {
                g.setColour(juce::Colours::black.withAlpha(0.75f));
                g.setFont(F(9.f, juce::Font::bold));
                g.drawText("C" + juce::String(n / 12 - 1), kr.toNearestInt().withTrimmedRight(3), juce::Justification::centredRight);
            }
        }

        // traces
        g.saveState();
        g.reduceClipRegion(plot.toNearestInt());
        const int N = BlackTune::kTrail;
        const int pos = proc.trailPos.load();
        juce::Path detP, outP;
        bool detOpen = false, outOpen = false;
        for (int i = 0; i < N; ++i)
        {
            const int idx = (pos + i) % N;
            const float x = plot.getX() + plot.getWidth() * (float) i / (float) (N - 1);
            const float dm = proc.trailDet[(size_t) idx].load();
            const float om = proc.trailOut[(size_t) idx].load();
            if (dm > 0.f)
            {
                const float y = yOf(dm);
                if (!detOpen) { detP.startNewSubPath(x, y); detOpen = true; } else detP.lineTo(x, y);
            }
            else detOpen = false;
            if (om > 0.f)
            {
                const float y = yOf(om);
                if (!outOpen) { outP.startNewSubPath(x, y); outOpen = true; } else outP.lineTo(x, y);
            }
            else outOpen = false;
        }
        g.setColour(juce::Colours::white.withAlpha(0.6f));
        g.strokePath(detP, juce::PathStrokeType(1.3f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        g.setColour(kGold.withAlpha(0.22f));
        g.strokePath(outP, juce::PathStrokeType(8.f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        g.setColour(juce::Colour(0xffffd45c));
        g.strokePath(outP, juce::PathStrokeType(2.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        g.setGradientFill(juce::ColourGradient(juce::Colour(0xff070707).withAlpha(0.95f), plot.getX(), 0.f,
                                               juce::Colours::transparentBlack, plot.getX() + 90.f, 0.f, false));
        g.fillRect(plot.withWidth(90.f));
        if (!tuneOn)
        {
            g.setColour(juce::Colours::black.withAlpha(0.45f));
            g.fillRect(plot);
        }
        // legend
        g.setFont(F(9.f, juce::Font::bold, 0.1f));
        const float lx = plot.getRight() - 128.f, ly = plot.getY() + 7.f;
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.drawLine(lx, ly + 5.f, lx + 14.f, ly + 5.f, 1.4f);
        g.drawText("INPUT", (int) lx + 18, (int) ly, 40, 10, juce::Justification::left);
        g.setColour(juce::Colour(0xffffd45c));
        g.drawLine(lx + 62.f, ly + 5.f, lx + 76.f, ly + 5.f, 2.4f);
        g.drawText("TUNED", (int) lx + 80, (int) ly, 40, 10, juce::Justification::left);
        g.restoreState();

        g.setColour(kGold.withAlpha(0.25f));
        g.drawRect(plot, 1.f);
        g.restoreState();
    }

    //==========================================================================
    // response-curve helpers
    static float freqToX(double f, double fmin, double fmax, juce::Rectangle<float> r)
    {
        return r.getX() + r.getWidth() * (float) (std::log(f / fmin) / std::log(fmax / fmin));
    }
    static float dbToY(double db, double lo, double hi, juce::Rectangle<float> r)
    {
        return r.getBottom() - r.getHeight() * (float) ((db - lo) / (hi - lo));
    }

    void drawGraphFrame(juce::Graphics& g, juce::Rectangle<float> r, double fmin, double fmax,
                        double dbLo, double dbHi, double dbStep, std::initializer_list<double> freqs)
    {
        g.setColour(juce::Colour(0xff070707));
        g.fillRoundedRectangle(r, 6.f);
        g.setColour(kGold.withAlpha(0.2f));
        g.drawRoundedRectangle(r, 6.f, 1.f);
        g.setFont(F(10.f, juce::Font::plain));

        for (double db = std::ceil(dbLo / dbStep) * dbStep; db <= dbHi + 1e-6; db += dbStep)
        {
            const float y = dbToY(db, dbLo, dbHi, r);
            g.setColour(juce::Colours::white.withAlpha(std::abs(db) < 1e-6 ? 0.28f : 0.07f));
            g.drawHorizontalLine((int) y, r.getX() + 1.f, r.getRight() - 1.f);
            g.setColour(juce::Colour(0xff6a6a6a));
            g.drawText((db > 0 ? "+" : "") + juce::String((int) std::round(db)),
                       (int) r.getX() + 4, (int) y - 12, 30, 12, juce::Justification::left);
        }
        for (double f : freqs)
        {
            const float x = freqToX(f, fmin, fmax, r);
            g.setColour(juce::Colours::white.withAlpha(0.07f));
            g.drawVerticalLine((int) x, r.getY() + 1.f, r.getBottom() - 1.f);
            g.setColour(juce::Colour(0xff6a6a6a));
            const juce::String lab = f >= 1000.0 ? juce::String((int) (f / 1000.0)) + "k" : juce::String((int) f);
            g.drawText(lab, (int) x - 15, (int) r.getBottom() - 14, 30, 12, juce::Justification::centred);
        }
    }

    template <typename DbFn>
    void drawCurve(juce::Graphics& g, juce::Rectangle<float> r, double fmin, double fmax,
                   double dbLo, double dbHi, bool on, DbFn dbAt)
    {
        juce::Path line;
        const int steps = juce::jmax(10, (int) (r.getWidth() / 2.f));
        for (int s = 0; s <= steps; ++s)
        {
            const double f = fmin * std::pow(fmax / fmin, (double) s / (double) steps);
            const double db = juce::jlimit(dbLo, dbHi, dbAt(f));
            const float x = r.getX() + r.getWidth() * (float) s / (float) steps;
            const float y = dbToY(db, dbLo, dbHi, r);
            if (s == 0) line.startNewSubPath(x, y); else line.lineTo(x, y);
        }
        juce::Path fill(line);
        const float y0 = dbToY(0.0, dbLo, dbHi, r);
        fill.lineTo(r.getRight(), y0);
        fill.lineTo(r.getX(), y0);
        fill.closeSubPath();

        g.saveState();
        g.reduceClipRegion(r.toNearestInt());
        g.setColour((on ? kGold : juce::Colour(0xff8a8a8a)).withAlpha(0.14f));
        g.fillPath(fill);
        g.setColour(on ? kGold : juce::Colour(0xff8a8a8a));
        g.strokePath(line, juce::PathStrokeType(2.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        g.restoreState();
    }

    void drawPageTitle(juce::Graphics& g, const char* title, const char* caption, bool on)
    {
        g.setColour(on ? kGold : juce::Colour(0xff8a8a8a));
        g.setFont(F(20.f, juce::Font::bold));
        g.drawText(title, 36, 130, 300, 26, juce::Justification::centredLeft);
        g.setColour(juce::Colour(0xff777777));
        g.setFont(F(11.f, juce::Font::plain));
        g.drawText(caption, 36, 154, 360, 14, juce::Justification::centredLeft);
    }

    void drawAirPage(juce::Graphics& g)
    {
        drawPanel(g, fxPanel, true);
        const bool on = proc.apvts.getRawParameterValue("air_on")->load() > 0.5f;
        drawPageTitle(g, "FRESH AIR", "Silky brightness and air for the voice. It eases off on harsh S sounds.", on);

        const auto r = airCurve.toFloat();
        drawGraphFrame(g, r, 200.0, 20000.0, -3.0, 12.0, 3.0, { 500, 1000, 2000, 5000, 10000 });

        const double amt  = proc.apvts.getRawParameterValue("air_amount")->load();
        const double freq = proc.apvts.getRawParameterValue("air_freq")->load();
        const double srd = 48000.0;
        Biquad sh;
        sh.setHighShelf(srd, freq * 0.8, amt * 8.0);
        drawCurve(g, r, 200.0, 20000.0, -3.0, 12.0, on, [&](double f)
        {
            return 20.0 * std::log10(std::max(1e-6, std::abs(sh.response(6.283185307179586 * f / srd))));
        });
        const float mx = freqToX(freq, 200.0, 20000.0, r);
        g.setColour(kGold.withAlpha(on ? 0.5f : 0.2f));
        g.drawVerticalLine((int) mx, r.getY() + 4.f, r.getBottom() - 16.f);
        g.setColour(juce::Colour(0xff9a9a9a));
        g.setFont(F(10.f, juce::Font::bold));
        g.drawText("SHINE", (int) mx - 25, (int) r.getY() + 4, 50, 12, juce::Justification::centred);

        // activity meter: how much air is being added right now
        const auto b = airBar.toFloat();
        g.setColour(juce::Colour(0xff9a9a9a));
        g.setFont(F(11.f, juce::Font::bold));
        g.drawText("AIR ACTIVITY", (int) b.getX(), (int) b.getY(), 110, (int) b.getHeight(), juce::Justification::centredLeft);
        const int segs = 40;
        const float segW = (b.getWidth() - 120.f) / (float) segs;
        const int lit = (int) std::round(proc.airActivity.load() * (float) segs);
        for (int i = 0; i < segs; ++i)
        {
            auto seg = juce::Rectangle<float>(b.getX() + 120.f + (float) i * segW + 1.f, b.getY() + 2.f, segW - 2.f, b.getHeight() - 4.f);
            const juce::Colour c = i < 26 ? kGold : kRed;
            g.setColour(i < lit && on ? c : juce::Colour(0xff191919));
            g.fillRoundedRectangle(seg, 2.f);
        }
    }

    void drawEqPage(juce::Graphics& g)
    {
        drawPanel(g, fxPanel, true);
        const bool on = proc.apvts.getRawParameterValue("eq_on")->load() > 0.5f;
        drawPageTitle(g, "EQUALIZER", "Low cut, four bands with adjustable frequency, and high cut.", on);

        const auto r = eqCurve.toFloat();
        drawGraphFrame(g, r, 20.0, 20000.0, -15.0, 15.0, 5.0, { 50, 100, 200, 500, 1000, 2000, 5000, 10000 });

        float v[10];
        for (int i = 0; i < 10; ++i) v[i] = proc.apvts.getRawParameterValue(kEqIds[i])->load();
        const double srd = 48000.0;
        Biquad hp, lo, lm, hm, hi, lp;
        hp.setHighPass(srd, v[0], 0.7071);
        lo.setLowShelf(srd, v[2], v[1]);
        lm.setPeak(srd, v[4], 0.9, v[3]);
        hm.setPeak(srd, v[6], 0.9, v[5]);
        hi.setHighShelf(srd, v[8], v[7]);
        lp.setLowPass(srd, v[9], 0.7071);
        const bool useHP = v[0] > 20.5f, useLP = v[9] < 19900.f;

        auto dbAt = [&](double f)
        {
            const double w = 6.283185307179586 * f / srd;
            std::complex<double> h = lo.response(w) * lm.response(w) * hm.response(w) * hi.response(w);
            if (useHP) h *= hp.response(w);
            if (useLP) h *= lp.response(w);
            return 20.0 * std::log10(std::max(1e-6, std::abs(h)));
        };
        drawCurve(g, r, 20.0, 20000.0, -15.0, 15.0, on, dbAt);

        // band handles on the curve
        auto dot = [&](double f, double db, juce::Colour c)
        {
            const float x = freqToX(f, 20.0, 20000.0, r);
            const float y = dbToY(juce::jlimit(-15.0, 15.0, db), -15.0, 15.0, r);
            g.setColour(c.withAlpha(on ? 0.28f : 0.12f));
            g.fillEllipse(x - 9.f, y - 9.f, 18.f, 18.f);
            g.setColour(on ? c : c.withAlpha(0.5f));
            g.fillEllipse(x - 5.f, y - 5.f, 10.f, 10.f);
            g.setColour(juce::Colours::black.withAlpha(0.8f));
            g.drawEllipse(x - 5.f, y - 5.f, 10.f, 10.f, 1.f);
        };
        if (useHP) dot(v[0], dbAt(v[0]), juce::Colour(0xff9a9a9a));
        if (useLP) dot(v[9], dbAt(v[9]), juce::Colour(0xff9a9a9a));
        dot(v[2], v[1], kBand[0]);
        dot(v[4], v[3], kBand[1]);
        dot(v[6], v[5], kBand[2]);
        dot(v[8], v[7], kBand[3]);

        // group captions above the knobs
        struct Grp { int a, b; const char* name; juce::Colour c; };
        const Grp groups[6] = { { 0, 0, "LOW CUT", juce::Colour(0xff9a9a9a) }, { 1, 2, "LOW", kBand[0] },
                                { 3, 4, "LOW MID", kBand[1] },               { 5, 6, "HIGH MID", kBand[2] },
                                { 7, 8, "HIGH", kBand[3] },                  { 9, 9, "HIGH CUT", juce::Colour(0xff9a9a9a) } };
        g.setFont(F(11.f, juce::Font::bold));
        for (const auto& gr : groups)
        {
            const int x = 36 + gr.a * 73 + 4;
            const int w = (gr.b - gr.a + 1) * 73 - 8;
            g.setColour(gr.c);
            g.drawText(gr.name, x, 338, w, 14, juce::Justification::centred);
            g.setColour(gr.c.withAlpha(0.6f));
            g.fillRect(x, 354, w, 2);
        }
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BlackTuneEditor)
};

juce::AudioProcessorEditor* BlackTune::createEditor() { return new BlackTuneEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new BlackTune(); }
