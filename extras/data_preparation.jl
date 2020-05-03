using DataFrames
using CSV

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

run(`bunzip2 ./data/2006.csv.bz2`)
run(`bunzip2 ./data/2007.csv.bz2`)
