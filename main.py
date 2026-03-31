mortalitas_df = pd.read_csv(
    "./data/mortalitas.csv", parse_dates=["Year"], date_format="%Y"
)
populasi_df = pd.read_csv("./data/populasi.csv")
bi_rate_df = pd.read_csv(
    "./data/bi_rate.csv", parse_dates=["Date"], date_format="%d-%m-%y"
)
