# setup --------------------------------------------
library(reticulate)
library(ggplot2)

# reticulate::virtualenv_create("nixtla-dev")
reticulate::use_virtualenv("nixtla-dev")

# dev version just released CP ! 
#reticulate::py_install("git+https://github.com/Nixtla/neuralforecast.git")

nf <- import("neuralforecast")
KAN <- nf$models$KAN
prediction_intervals <- nf$utils$PredictionIntervals(method="conformal_error")
RMSE <- nf$losses$pytorch$RMSE

# R wrappers ------------------------------------

kan_spec <- \(h = 12L, input_size = 24L, max_steps=20L){
   nf$NeuralForecast(
    models = list(
      KAN(h=h,
          input_size=input_size,
          loss=RMSE(),
          max_steps=max_steps,
          scaler_type='standard'
      )
    ),
    freq='ME'
  )
  
}

conformal_fit <- \(model_spec = NULL, df = NULL){
  model_spec$fit(df = df, prediction_intervals = prediction_intervals)
}

conformal_predict <- \(model_spec = NULL, model_fit = NULL, level = NULL){
  model_spec$predict(model_fit, level = list(level))
}

# get data ------------

# think a true R binding of nixlta-kan to say modeltime should be for one unique_id, 
# then let modeltime handle grouping or map(), 
# avoiding unique ID complications 
# since the pandas row ids aren't carried to R, avoid rep(.., h) confusion

air_passengers_df <- readr::read_csv("https://raw.githubusercontent.com/frankiethull/nixtla-r-tutorial/refs/heads/main/airpassengersDF.csv")

train_df <- air_passengers_df |> dplyr::slice(1:132)
test_df  <- air_passengers_df |> dplyr::anti_join(train_df)

# setup, train, predict -------------

# specs
kan_model_specs <- kan_spec(h = 12L)

# fit
kan_model_fit   <- conformal_fit(model_spec = kan_model_specs, df = train_df)

# predict
conf_kan_preds  <- conformal_predict(model_spec = kan_model_specs, 
                                     model_fit = kan_model_fit,
                                     level = 90L)


# data vizzing ---------------------------------------------------------------

gg_datasplit <- conf_kan_preds |>
  ggplot() +
  geom_ribbon(aes(x = ds, ymin = `KAN-lo-90`, ymax = `KAN-hi-90`), 
              fill = "darkcyan", alpha = .4) + 
  geom_line(aes(x = ds, y = KAN, color = "KAN")) + 
  geom_line(data = train_df, aes(x = ds, y = y, color = "Training")) +
  geom_line(data = test_df,  aes(x = ds, y = y, color = "Testing")) + 
  geom_vline(data = NULL, aes(xintercept = max(train_df$ds)), linetype = 2, color = "grey30") + 
  ggthemes::theme_fivethirtyeight() +
  scale_color_manual(values = c("darkcyan", "orange", "midnightblue")) +
  theme(legend.title = element_blank()) +
  labs(subtitle = "Training/Test Data")

gg_test_ts <- conf_kan_preds |>
  ggplot() +
  geom_ribbon(aes(x = ds, ymin = `KAN-lo-90`, ymax = `KAN-hi-90`), 
              fill = "darkcyan", alpha = .4) + 
  geom_line(aes(x = ds, y = KAN), color = "darkcyan") +
  geom_line(data = test_df,  aes(x = ds, y = y), color = "orange") +
  ggthemes::theme_fivethirtyeight() +
  labs(subtitle = "Conformal KAN vs Test")


gt_test <- conf_kan_preds |> 
  cbind(test_df |> dplyr::select(y)) |>
  dplyr::mutate(ds = as.Date(ds)) |> 
  gt::gt() |>
  gt::fmt_number(columns = -ds, decimals = 0) |>
  gt::data_color(columns = -ds, palette = c("darkcyan", "orange")) |>
  gt::tab_header(title = "", subtitle = "Conformal KAN vs Actual") |>
  gtUtils::gt_theme_sofa()
  
# dashboarding ------------------------------------  
library(patchwork)

layout <- "
AAC
AAC
BBD
"

ggkan_patch <-
gg_test_ts + gg_datasplit + 
   gt_test + guide_area() + 
  plot_layout(design = layout, guides = "collect") + 
  plot_annotation(title = "Conformal KAN via {neuralforecast}",
                  subtitle = "Nixtla's Kolmogorov Arnold Networks in R with Conformal Prediction Intervals",
                  caption = "Kolmogorov-Arnold Networks (KANs) are an\nalternative to Multi-Layer Perceptrons (MLPs).\nusing {neuralforecast} via {reticulate}") &
  ggthemes::theme_fivethirtyeight() &
  theme(legend.title = element_blank())

  

ggsave(filename = r"(C:\Users\Frank\Documents\R Scripts\nixtla\KAN\ggkan_patch.png)",
       plot = ggkan_patch,
       height = 6,
       width = 10,
       dpi = 300)
