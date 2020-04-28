using Documenter, StrayCopse

makedocs(;
    modules=[StrayCopse],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Arkoniak/StrayCopse.jl/blob/{commit}{path}#L{line}",
    sitename="StrayCopse.jl",
    authors="Andrey Oskin",
    assets=String[],
)

deploydocs(;
    repo="github.com/Arkoniak/StrayCopse.jl",
)
