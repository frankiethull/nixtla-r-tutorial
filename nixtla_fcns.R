# 00 importer ----
import_neuralforecast <- function(){
  nf <- reticulate::import("neuralforecast")
  
  return(nf)
}

# initiate neuralforecast 
nf <- import_neuralforecast()

# 01 lazy aliasing for models & losses ---- 
NBEATS  <- nf$models$NBEATS
NBEATSx <- nf$models$NBEATSx
NHITS   <- nf$models$NHITS
LSTM    <- nf$models$LSTM
RNN     <- nf$models$RNN

MQLoss  <- nf$losses$pytorch$MQLoss


# 02 model workflow ----

# staging fcn
neural_model_setup <- function(models = models, frequency = "M"){
  nf$NeuralForecast(models = models, freq = frequency)
}

# fitting fcn
neural_model_fit <- function(model_setup = neural_model_setup, df = Y_train_df){
  model_setup$fit(df = df)
}

# forecasting fcn
neural_model_predict <- function(model_setup = neural_model_setup, model_fit = neural_model_fit){
  model_setup$predict(model_fit)
}
