// VoxViper - aggressive hard-tune / robot vocal effect
// Pipeline: YIN pitch detection -> snap to scale -> delay-line pitch shifter
//           -> pitched comb filter (robot) -> drive + bitcrush -> dry/wet mix
#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <cmath>
#include <vector>

class VoxViper : public juce::AudioProcessor
{
public:
    VoxViper()
        : AudioProcessor(BusesProperties()
                             .withInput("Input", juce::AudioChannelSet::stereo(), true)
                             .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "PARAMS", createLayout())
    {
    }

    const juce::String getName() const override { return "VoxViper"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override
    {
        return new juce::GenericAudioProcessorEditor(*this);
    }

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

    //==========================================================================
    void prepareToPlay(double sampleRate, int) override
    {
        sr = sampleRate;
        W = (int) std::round(0.023 * sr);           // shifter window (~23 ms)
        setLatencySamples(W / 2);

        for (auto& v : dl)   v.assign(kBuf, 0.f);
        for (auto& v : comb) v.assign(kBuf, 0.f);
        det.assign(kBuf, 0.f);
        scratch.assign((size_t) (kBuf), 0.f);
        yin.assign((size_t) (kBuf), 0.f);

        wp = 0; hopCount = 0; hop = 512;
        phase = 0.f; curRatio = 1.f; targetRatio = 1.f;
        targetFreq = 200.f; combT = (float) (sr / 200.0);
    }
    void releaseResources() override {}

    //==========================================================================
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

            // advance shifter phase
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

                // robot: pitched feedback comb
                float yc = wet + fb * readFrac(comb[(size_t) c], combT);
                comb[(size_t) c][(size_t) (wp & kMask)] = yc;
                wet = yc * combNorm;

                // grit: drive + bitcrush
                wet = std::tanh(wet * drive);
                wet = std::round(wet * levels) / levels;

                const float dry = dl[(size_t) c][(size_t) ((wp - W / 2) & kMask)];
                io[i] = dry * (1.f - mix) + wet * mix;
            }
            ++wp;
        }
    }

private:
    static constexpr int kBuf = 1 << 14;
    static constexpr int kMask = kBuf - 1;

    juce::AudioProcessorValueTreeState apvts;
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
        l.add(std::make_unique<AudioParameterChoice>(ParameterID{"key", 1}, "Key",
              StringArray{"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"}, 0));
        l.add(std::make_unique<AudioParameterChoice>(ParameterID{"scale", 1}, "Scale",
              StringArray{"Chromatic","Major","Minor","Minor Pentatonic","Harmonic Minor"}, 2));
        l.add(std::make_unique<AudioParameterFloat>(ParameterID{"retune", 1}, "Retune Speed (ms)",
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

    // YIN pitch detection + scale quantisation
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
        if (std::sqrt(energy / (float) N) < 0.004f) { targetRatio = 1.f; return; }

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
        if (found < 0) { targetRatio = 1.f; return; }

        float tauEst = (float) found;
        const float s0 = yin[(size_t) found - 1], s1 = yin[(size_t) found], s2 = yin[(size_t) found + 1];
        const float den = s0 - 2.f * s1 + s2;
        if (std::abs(den) > 1e-9f) tauEst += 0.5f * (s0 - s2) / den;

        const float f0 = (float) sr / tauEst;
        const float midi = 69.f + 12.f * std::log2(f0 / 440.f);

        static const int masks[5] = { 0xFFF, 0xAB5, 0x5AD, 0x4A9, 0x9AD };
        // bit k set => semitone k above the key is allowed
        const int mask = masks[juce::jlimit(0, 4, scaleIdx)];

        float best = midi; float bestDist = 1e9f;
        const int base = (int) std::round(midi);
        for (int o = -6; o <= 6; ++o)
        {
            const int note = base + o;
            const int rel = ((note - keyIdx) % 12 + 12) % 12;
            if (!(mask & (1 << rel))) continue;
            const float dist = std::abs((float) note - midi);
            if (dist < bestDist) { bestDist = dist; best = (float) note; }
        }

        targetRatio = std::pow(2.f, ((best - midi) * amountP) / 12.f);
        targetFreq = 440.f * std::pow(2.f, ((midi + (best - midi) * amountP) - 69.f) / 12.f);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VoxViper)
};

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new VoxViper(); }
