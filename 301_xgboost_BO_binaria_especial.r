#EN CONSTRUCCION

#Mi nueva clase binaria  es  1 = { BAJA+2, BAJA+1 }
#usa el dataset EXTENDIDO, que tiene Feature Engineering
#prueba con prob_corte entre 0.025 y 0.050 porque ahora los "positivos"  son más
#Cross Validation

#limpio la memoria
rm( list=ls() )
gc()

require("data.table")
require("rlist")
require("yaml")

require("xgboost")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

?mbo
#defino la carpeta donde trabajo
setwd("C:/Users/pbeltramino/Desktop/MGCD/Laboratorio de Implementacion I" )

kfinalize <- FALSE
kexperimento  <- 6084

kscript           <- "xgboost_BO_binaria_especial"
karch_generacion  <- "./datasets/corregido_201905_ext_tarjetas_financieras_20210412.csv"
karch_aplicacion  <- "./datasets/corregido_201907_ext_tarjetas_financieras_20210412.csv"
kBO_iter    <-  100   #cantidad de iteraciones de la Optimizacion Bayesiana

hs <- makeParamSet( 
         makeNumericParam("eta",              lower= 0.01 , upper=    0.1),
         makeNumericParam("colsample_bytree", lower= 0.2  , upper=    1.0),
         makeNumericParam("alpha",            lower= 0    , upper=    5),
         makeNumericParam("lambda",           lower= 0    , upper=  200),
         makeNumericParam("gamma",            lower= 0    , upper=    5),
         makeNumericParam("min_child_weight", lower= 0    , upper=   20),
         makeIntegerParam("max_leaves",       lower=32L   , upper= 1024L),
         makeNumericParam("prob_corte",       lower= 0.025, upper=    0.050),  #aumento la probabilidad de corte
         makeNumericParam("max_depth",        lower= 0,     upper=    0.1),
         makeNumericParam("max_bin",          lower=31,     upper=   31.1),
         makeNumericParam("early_stopping_rounds", lower=200, upper=201 )
        )

#------------------------------------------------------------------------------

get_experimento  <- function()
{
  if( !file.exists( "./maestro.yaml" ) )  cat( file="./maestro.yaml", "experimento: 1000" )
  
  exp  <- read_yaml( "./maestro.yaml" )
  experimento_actual  <- exp$experimento
  
  exp$experimento  <- as.integer(exp$experimento + 1)
  Sys.chmod( "./maestro.yaml", mode = "0644", use_umask = TRUE)
  write_yaml( exp, "./maestro.yaml" )
  Sys.chmod( "./maestro.yaml", mode = "0444", use_umask = TRUE) #dejo el archivo readonly
  
  return( experimento_actual )
}
#------------------------------------------------------------------------------

if( is.na(kexperimento ) )   kexperimento <- get_experimento()  #creo el experimento

#en estos archivos queda el resultado
kbayesiana  <- paste0("./work/E",  kexperimento, "_xgboost.RDATA" )
klog        <- paste0("./work/E",  kexperimento, "_xgboost_log.txt" )
kimp        <- paste0("./work/E",  kexperimento, "_xgboost_importance_" )
kmbo        <- paste0("./work/E",  kexperimento, "_xgboost_mbo.txt" )
kmejor      <- paste0("./work/E",  kexperimento, "_xgboost_mejor.yaml" )
kkaggle     <- paste0("./kaggle/E",kexperimento, "_xgboost_kaggle_" )

#------------------------------------------------------------------------------

loguear  <- function( reg, pscript, parch_generacion, arch)
{
  if( !file.exists(  arch ) )
  {
    linea  <- paste0( "script\tdataset\tfecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )

    cat( linea, file=arch )
  }

  linea  <- paste0( pscript, "\t",
                    parch_generacion, "\t",
                    format(Sys.time(), "%Y%m%d %H%M%S"),
                    "\t",
                    gsub( ", ", "\t", toString( reg ) ),
                    "\n" )

  cat( linea, file=arch, append=TRUE )
}
#------------------------------------------------------------------------------

SCORE_PCORTE <-  log( 0.025 / ( 1 - 0.025 ) )   #esto hace falta en ESTA version del XGBoost ... misterio por ahora ...

fganancia_logistic_xgboost   <- function(scores, clases) 
{
   vlabels  <- getinfo(clases, "label")
   vpesos   <- getinfo(clases, "weight")  #truco
  
   gan  <- sum( (scores > SCORE_PCORTE  ) * 
                ifelse( vlabels== 1 & vpesos>1, 29250, -750 )   
             )

   return(  list("metric" = "ganancia", "value" = gan ) )
}
#------------------------------------------------------------------------------
#funcion que va a optimizar la Bayesian Optimization

EstimarGananciaXGBoostCV  <- function( x )
{
  SCORE_PCORTE  <<- log( x$prob_corte / ( 1 - x$prob_corte ) )  

  kfolds  <- 5   # cantidad de folds para cross validation

  set.seed( 999983 )
  modelocv  <- xgb.cv( data= dtrain,
                       objective= "binary:logistic",
                       tree_method= "hist",
                       grow_policy= "lossguide",
                       stratified= TRUE, #sobre el cross validation
                       nfold = kfolds,   #folds del cross validation
                       feval= fganancia_logistic_xgboost,
                       maximize= TRUE,
                       disable_default_eval_metric= TRUE,
                       base_score= mean( getinfo(dtrain, "label")),
                       early_stopping_rounds= x$early_stopping_rounds,  #deberia agrandarse
                       nround= 5000,   #un numero muy grande
                       max_bin= as.integer(x$max_bin),
                       max_depth= as.integer(x$max_depth),
                       eta= x$eta,
                       colsample_bytree= x$colsample_bytree,
                       max_leaves= x$max_leaves,
                       min_child_weight= x$min_child_weight,
                       alpha= x$alpha, 
                       lambda= x$lambda, 
                       gamma= x$gamma,
                       outputmargin= FALSE,
                       nthread= 0,  #cantidad de nucleos del procesador que se utilizan
                       verbose= FALSE
                     )


  mejor_iter  <- modelocv$best_iter
  ganancia    <- unlist( modelocv$evaluation_log[ , test_ganancia_mean] )[ mejor_iter ] 

  ganancia_normalizada  <-  ganancia* kfolds 
  attr(ganancia_normalizada ,"extras" )  <- list("pnround"= modelocv$best_iter)  #esta es la forma de devolver un parametro extra

  xx  <- x
  xx$best_iter  <-  modelocv$best_iter
  xx$ganancia   <-  ganancia_normalizada
  loguear( xx, kscript, karch_generacion, klog )

  return( ganancia_normalizada )
}
#------------------------------------------------------------------------------

#cargo el dataset
dataset  <- fread(karch_generacion)

#creo la clase_binaria especial  1 = { BAJA+2, BAJA+1 }
dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]


campos_buenos  <- setdiff( colnames(dataset) , c("clase_ternaria", "clase01") )

#genero el formato requerido por XGBoost
#agrego peso a cada registro como un MUY SUCIO TRUCO de poder saber cual registro era originalmente BAJA+2
#como es apenas mayor a 1, no va a tener efecto en el algoritmo
dtrain  <- xgb.DMatrix( data = data.matrix(  dataset[ , campos_buenos, with=FALSE]),
                        label= dataset[ , clase01],
                        weight=  dataset[ , ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)]
                      )


#Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar <-  EstimarGananciaXGBoostCV

configureMlr(show.learner.output = FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
obj.fun  <- makeSingleObjectiveFunction(
              fn= funcion_optimizar,
              minimize= FALSE,   #estoy Maximizando la ganancia
              noisy=    TRUE,
              par.set= hs,   #los hiperparametros que quiero optimizar, definidos al inicio del script
              has.simple.signature= FALSE
             )


ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI())

surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace = TRUE))


if( kfinalize )
{
  mboFinalize(kbayesiana)
}

#Esta es la corrida en serio
if( !file.exists(kbayesiana) )
{ 
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl) 
} else {
  run  <- mboContinue( kbayesiana )
}

?mbo
#ordeno las corridas
tbl  <- as.data.table(run$opt.path)
tbl[ , iteracion := .I ]  #le pego el numero de iteracion
setorder( tbl, -y )

#agrego info que me viene bien
tbl[ , script          := kscript ]
tbl[ , arch_generacion := karch_generacion ]
tbl[ , arch_aplicacion := karch_aplicacion ]

fwrite(  tbl, file=kmbo, sep="\t" )   #grabo TODA la corrida
write_yaml( tbl[1], kmejor )          #grabo el mejor

#------------------------------------------------------------------------------
#genero las mejores 5 salidas para Kaggle

#cargo los datos de 201907, que es donde voy a APLICAR el modelo
dataset_aplicar  <- fread(karch_aplicacion)
#genero el dataset en formato que necesita XGBoost
dapply  <- xgb.DMatrix( data = data.matrix(  dataset_aplicar[ , campos_buenos, with=FALSE]) )

for( modelito in  1:5 )
{
  x  <- tbl[modelito]   #en x quedaron los MEJORES hiperparametros

  set.seed( 999983 )
  modelo  <- xgboost( data= dtrain,
                      objective= "binary:logistic",
                      tree_method= "hist",
                      grow_policy= "lossguide",
                      max_depth= as.integer(x$max_depth),
                      base_score= mean( getinfo(dtrain, "label")),
                      nround= x$pnround,   #MUY IMPORTANTE
                      max_bin= as.integer(x$max_bin),
                      eta= x$eta,
                      colsample_bytree= x$colsample_bytree,
                      max_leaves= x$max_leaves,
                      min_child_weight= x$min_child_weight,
                      alpha= x$alpha, 
                      lambda= x$lambda, 
                      gamma= x$gamma,
                      nthread= 0  #cantidad de nucleos del procesador que se utilizan
                    )

  #importancia de variables
  tb_importancia  <- xgb.importance( model= modelo )
  fwrite( tb_importancia, 
          file= paste0(kimp,  modelito,".txt"),
          sep="\t" )


  #genero el vector con la prediccion, la probabilidad de ser positivo
  prediccion  <- predict( modelo, dapply)

  dataset_aplicar[ , prob_pos := prediccion]
  dataset_aplicar[ , estimulo := as.numeric(prob_pos > x$prob_corte) ]

  entrega  <- dataset_aplicar[   , list( numero_de_cliente, estimulo)  ]

  #genero el archivo para Kaggle
  fwrite( entrega, 
          file= paste0(kkaggle, modelito, ".csv"),
          sep=  "," )
}

quit( save="no" )


