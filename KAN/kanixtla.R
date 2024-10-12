library(reticulate)
library(ggplot2)

# reticulate::virtualenv_create("nixtlakan")
reticulate::use_virtualenv("nixtlakan")
# reticulate::py_install("neuralforecast", pip = TRUE)

nf <- import("neuralforecast")
KAN <- nf$models$KAN

DistributionLoss <- nf$losses$pytorch$DistributionLoss

AirPassengersPanel <- nf$utils$AirPassengersPanel
AirPassengersStatic <- nf$utils$AirPassengersStatic

# Python code adapted for R
Y_train_df <- AirPassengersPanel |> dplyr::group_by(unique_id) |> dplyr::slice(1:120)
Y_test_df  <- AirPassengersPanel |> dplyr::anti_join(Y_train_df)

# Initialize and fit the model
h=24L
fcst <- nf$NeuralForecast(
  models = list(
    KAN(h=h,
        input_size=24L,
        loss=DistributionLoss(distribution="Normal"),
        max_steps=100L,
        scaler_type='standard',
        futr_exog_list=list('y_[lag12]'),
        hist_exog_list=NULL,
        stat_exog_list=list('airline1')
    )
  ),
  freq='M'
)

fcst$fit(df=Y_train_df, static_df=AirPassengersStatic)

# Predict
forecasts <- fcst$predict(futr_df=Y_test_df)

forecasts <- 
forecasts |> 
  cbind(unique_id = c(rep("Airline1", h),rep("Airline2", h)))

ggkan <- 
AirPassengersPanel |>
  dplyr::filter(lubridate::year(ds) > 1957) |>
  ggplot() +
  geom_ribbon(data = forecasts,
              aes(x = ds, 
                  ymin = `KAN-lo-90`, ymax = `KAN-hi-90`), fill = "midnightblue", alpha = .4) + 
  geom_line(aes(x = ds, y = y, color = "Actuals")) +
  geom_line(data = forecasts,
            aes(x = ds, y = KAN, color = "KAN")) + 
  facet_wrap(~unique_id, nrow = 2, scales = "free") +
  theme_minimal() +
  ggthemes::theme_fivethirtyeight() +
  theme(legend.position = "top", legend.justification = "left",
        legend.title = element_blank()) +
  scale_color_manual(values = c("limegreen", "midnightblue")) +
  labs(title = "KAN via {neuralforecast}",
       subtitle = "Nixtla's Kolmogorov Arnold Networks in R",
       caption = "Kolmogorov-Arnold Networks (KANs) are an\nalternative to Multi-Layer Perceptrons (MLPs).\nusing {neuralforecast} via {reticulate}")


ggsave(filename = r"(C:\Users\Frank\Documents\R Scripts\nixtla\KAN\ggkan.png)",
       plot = ggkan,
       height = 5,
       width = 6,
       dpi = 300)
