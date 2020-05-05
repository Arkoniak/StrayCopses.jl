using DataFrames
using CSV
using StatsBase
using StableRNGs

# https://github.com/szilard/benchm-ml/blob/master/0-init/2-gendata.txt
# 2005 2006 2007

# 2005
url2005 = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/JTFT25"

# 2006
url2006 = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/EPIFFT"

# 2007
url2007 = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HG7NV7/2BHLWK"

download(url2005, joinpath(@__DIR__, "data", "2005.csv.bz2"))
download(url2006, joinpath(@__DIR__, "data", "2006.csv.bz2"))
download(url2007, joinpath(@__DIR__, "data", "2007.csv.bz2"))

run(`bunzip2 ./data/2005.csv.bz2`)
run(`bunzip2 ./data/2006.csv.bz2`)
run(`bunzip2 ./data/2007.csv.bz2`)

df1 = CSV.file(joinpath(@__DIR__, "data", "2005.csv")) |> DataFrame;
df2 = CSV.file(joinpath(@__DIR__, "data", "2006.csv")) |> DataFrame;

df1 = df1[df1[!, :DepDelay] .!= "NA", :];
df2 = df2[df2[!, :DepDelay] .!= "NA", :];

df1[!, :dep_delayed_15min] = ifelse.(parse.(Int, df1[!, :DepDelay]) .>= 15, "Y", "N");
df2[!, :dep_delayed_15min] = ifelse.(parse.(Int, df2[!, :DepDelay]) .>= 15, "Y", "N");

cols = Symbol.(["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance","dep_delayed_15min"])

df1 = df1[!, cols];
df2 = df2[!, cols];

df = vcat(df1, df2);

for k in Symbol.(["Month", "DayofMonth", "DayOfWeek"])
    df[!, k] .= "c-" .* string.(df[!, k])
end

df[!, :DepTime] .= parse.(Int, df[!, :DepTime]);

CSV.write(joinpath(@__DIR__, "data", "air-process.csv"), df)

# Sampling

rng = StableRNG(2020)

ids1k = sample(rng, 1:size(df, 1), 1000, replace = false);
ids10k = sample(rng, 1:size(df, 1), 10000, replace = false);

df1k = df[ids1k, :];
df10k = df[ids10k, :];

CSV.write(joinpath(@__DIR__, "data", "air-1k.csv"), df1k);
CSV.write(joinpath(@__DIR__, "data", "air-10k.csv"), df10k);

###################
test data

df = urldownload("https://s3.amazonaws.com/benchm-ml--main/test.csv", true) |> DataFrame
CSV.write(joinpath(@__DIR__, "data", "air-test.csv"), df)

df = urldownload("https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv", true) |> DataFrame
CSV.write(joinpath(@__DIR__, "data", "air-0.1m.csv"), df)

df = urldownload("https://s3.amazonaws.com/benchm-ml--main/train-1m.csv", true) |> DataFrame
CSV.write(joinpath(@__DIR__, "data", "air-1m.csv"), df)

# optional
df = urldownload("https://s3.amazonaws.com/benchm-ml--main/train-10m.csv", true) |> DataFrame
CSV.write(joinpath(@__DIR__, "data", "air-10m.csv"), df)
