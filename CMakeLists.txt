cmake_minimum_required(VERSION 3.22)
project(VoxViper VERSION 1.0.0)

include(FetchContent)
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(VoxViper
    COMPANY_NAME "VoxLab"
    PLUGIN_MANUFACTURER_CODE Vxlb
    PLUGIN_CODE Vxvp
    FORMATS VST3 Standalone
    PRODUCT_NAME "VoxViper"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE)

target_sources(VoxViper PRIVATE Source/Plugin.cpp)

target_compile_definitions(VoxViper PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(VoxViper
    PRIVATE
        juce::juce_audio_utils
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
