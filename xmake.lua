add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

set_encodings("utf-8")

set_languages("c++20", "c11")

if is_plat("wasm") then
    add_requires("emscripten")
    set_toolchains("emcc@emscripten")
    add_ldflags("-sASSERTIONS=2", "-sDEMANGLE_SUPPORT=1", "-sEXPORTED_RUNTIME_METHODS=['FS']")
end

if is_plat("wasm") and is_arch("wasm64") then
    add_cxflags("-sMEMORY64=1")
    add_ldflags("-sMEMORY64=1", "-sWASM_BIGINT=1")
    add_ldflags("-sINITIAL_MEMORY=1073741824", "-sMAXIMUM_MEMORY=17179869184")
end

if is_plat("wasm") then
    add_requires("emscripten")
    set_toolchains("emcc@emscripten")
    add_ldflags("-sASSERTIONS=2", "-sDEMANGLE_SUPPORT=1", "-sEXPORTED_RUNTIME_METHODS=['FS']")
end

if is_plat("wasm") and is_arch("wasm64") then
    add_cxflags("-sMEMORY64=1")
    add_ldflags("-sMEMORY64=1", "-sWASM_BIGINT=1")
    add_ldflags("-sINITIAL_MEMORY=1073741824", "-sMAXIMUM_MEMORY=17179869184")
end

if is_plat("windows") then
    add_defines("NOMINMAX")

    add_cxflags("/utf-8")
    add_cxxflags("/utf-8")
end

if is_plat("windows", "mingw") then
    add_syslinks("user32", "gdi32")

    add_cxflags("/utf-8")
    add_cxxflags("/utf-8")
end

if is_plat("windows", "mingw") then
    add_syslinks("user32", "gdi32")
end

add_requires("ncnn master", {
    configs = {
        vulkan=true
    }
})

add_requires("nlohmann_json")
add_requires("nlohmann_json")

add_includedirs("src/")

target("ncnn_tokenizer")
    set_kind("static")
    add_files("src/utils/tokenizer/*.cpp")

target("ncnn_llm")
    set_kind("static")
    add_files("src/*.cpp")
    add_files("src/utils/*.cpp")
    add_files("src/utils/audio/*.cpp")
    add_deps("ncnn_tokenizer")
    add_packages("ncnn", "nlohmann_json")

target("llm_ncnn_run")
    set_kind("binary")
    add_includedirs("examples/")
    add_files("examples/llm_ncnn_run/*.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")

    set_rundir("$(projectdir)/")

target("benchllm")
    set_kind("binary")
    add_files("benchmark/benchllm.cpp")

    add_deps("ncnn_llm")
    add_packages("ncnn")

    set_rundir("$(projectdir)/assets/minicpm4_0.5b/")

target("test_llm")
    set_kind("binary")
    add_includedirs("tests/")
    add_files("tests/test_llm.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")

    set_rundir("$(projectdir)/")

target("nllb_main")
    set_kind("binary")
    add_files("examples/nllb_main.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn")

    set_rundir("$(projectdir)/")

target("embedding_main")
    set_kind("binary")
    add_files("examples/embedding_main.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")

    set_rundir("$(projectdir)/")

target("clip_main")
    set_kind("binary")
    add_files("examples/clip_main.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")

    set_rundir("$(projectdir)/")

target("ocr_main")
    set_kind("binary")
    add_files("examples/ocr_main.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")

    set_rundir("$(projectdir)/")

target("tts_main")
    set_kind("binary")
    add_files("examples/tts_main.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")

    set_rundir("$(projectdir)/")