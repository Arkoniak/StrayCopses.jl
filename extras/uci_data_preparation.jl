# This is for comparison with Leo Breiman paper "RANDOM FORESTS"
using UrlDownload
using DataFrames
using CSV
using StableRNGs
using StatsBase

datasets = [
   # "glass" dataset
   (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    kwargs = [:header => ["Id", "RI", "Na","Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glass_type"]],
    transform = x -> select!(x, Not(1)),
    out = "uci_glass",
    split = true),

   # "sonar" dataset
   (url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
    kwargs = [:header => false],
    transform = identity,
    out = "uci_sonar",
    split = true),

   # "image" dataset
   (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data",
      kwargs = [:skipto => 6, :header => replace.(lowercase.(["target", "REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN", "HUE-MEAN"]), "-" => "_")],
      transform = x -> select!(x, [names(x)[2:end]..., names(x)[1]]),
      out = "uci_images_train.csv",
      split = false
   ),

   (url = "http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test",
      kwargs = [:skipto => 6, :header => replace.(lowercase.(["target", "REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT", "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN", "HUE-MEAN"]), "-" => "_")],
      transform = x -> select!(x, [names(x)[2:end]..., names(x)[1]]),
      out = "uci_images_test.csv",
      split = false
   ),

   # "sat-images" dataset
   (url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn",
      kwargs = [:header => false, :delim => ' '],
      transform = identity,
      out = "uci_satimage_train.csv",
      split = false),

   (url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst",
      kwargs = [:header => false, :delim => ' '],
      transform = identity,
      out = "uci_satimage_test.csv",
      split = false),

   # "letters" dataset
   (url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
      kwargs = [:header => false],
      transform = x -> select!(x, [names(x)[2:end]..., names(x)[1]]),
      out = "uci_letters",
      split = true),

   # "iris" dataset
   (url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
      kwargs = [:header => false],
      transform = identity,
      out = "uci_iris_all.csv",
      split = false
   )
]

function load_dataset(dataset)
   df = urldownload(dataset.url,
      format = :CSV; dataset.kwargs...) |> DataFrame
   dataset.transform(df)
   if dataset.split
      rng = StableRNG(2020)
      test_ids = sample(rng, 1:size(df, 1), Int(floor(0.1*size(df, 1))), replace = false)
      train_ids = setdiff(1:size(df, 1), test_ids)
      CSV.write(joinpath(@__DIR__, "data", dataset.out*"_test.csv"), df[test_ids, :])
      CSV.write(joinpath(@__DIR__, "data", dataset.out*"_train.csv"), df[train_ids, :])
   else
      CSV.write(joinpath(@__DIR__, "data", dataset.out), df)
   end
end

for dataset in datasets
   load_dataset(dataset)
end

####
# letter recognition
load_dataset(datasets[end])
