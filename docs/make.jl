using Documenter, StrayCopses

makedocs(;
    modules=[StrayCopses],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Arkoniak/StrayCopses.jl/blob/{commit}{path}#L{line}",
    sitename="StrayCopses.jl",
    authors="Andrey Oskin"
)

deploydocs(;
    repo="github.com/Arkoniak/StrayCopses.jl",
)
