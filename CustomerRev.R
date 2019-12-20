#Groupe : IBO DataScience 2

#Johann de Soyres
#Olivier Dupain
#Henri de Montalembert
#Mathieu Josserand

#Projet - Apprentissage - 

memory.limit(size=56000)

#----------------------------Packages------------------------------------

require(jsonlite)
require(data.table)
require(dummies)
require(caTools)
require(naniar)
require(rpart)
require(rpart.plot)
require(dplyr)
require(hydroGOF)
require(randomForest)
require(ggplot2)
require(cowplot)
require(dismo)
require(doParallel)
require(gbm)

#------------------------Import des données------------------------------


#Lecture des données. Ici on se servira pas du fichier test.csv puisqu'il ne contient pas notre output (transactionRevenue).
fullDataset=fread("/Users/casden/Documents/S09_ESILV/Machine learning/projet/old data/train.csv",stringsAsFactors = FALSE,colClasses=c("character","integer","character","character","character","character","character","character","character","integer","integer","integer"),header=TRUE)
fullDataset=as.data.frame(fullDataset,stringsAsFactors=FALSE)
str(fullDataset)

char1=names(fullDataset)[which(sapply(fullDataset, class)%in% c("character"))]#on recupere les noms des col character
for(col in char1)
{
  fullDataset[[col]]=gsub("\"\"","\"",fullDataset[[col]])#on enleve les doubles guillemets
}

#Selection au hasard d'un echantillon des donnees
dataset=fullDataset[sample(nrow(fullDataset), 100000), ]

# création d'une colonne indicatrice train test avant assemblage des deux tables
dataset$datasplit<-"train" 

# suppression d'une colonne visiblement inutile
dataset$campaignCode<-NULL

# identification des 4 colonnes au format json
json<-c("trafficSource","totals","geoNetwork","device")

glob<-data.table() #table vide qui va récupérer les tables transformées

# lecture et transformation successive du dataset (suppression au passage de colonnes inutiles) 

partiel<-dataset[,setdiff(names(dataset),json)] # colonnes non json
for (j in json) partiel<-cbind(partiel,fromJSON(paste("[", paste(dataset[[j]], collapse = ","), "]")))
temp<-partiel$adwordsClickInfo
partiel$adwordsClickInfo<-NULL
temp$targetingCriteria<-NULL
result<-as.data.table(cbind(partiel,temp))
glob<-rbind(glob,result,fill=TRUE)
str(glob)

rm(partiel, dataset)
gc()#Suppresion des ressources memoires inutiles

#------------------------Traitement des données------------------------------

#On enleve les colonnes ID(sauf la colonne fullvisitorID) et la colonne date
glob = glob[, -c("date","sessionId","cityId","visitId","gclId")]

#On eneleve les colonnes qui comportent trop de NA (>50%).
gg_miss_var(glob)
glob = glob[, -c("referralPath","campaign","newVisits","keyword","adNetworkType","isVideoAd","page","slot","adContent")]

#on enleve les colonnes qui ont moins de 3 valeurs differentes, sauf pour les colonnes de type boolean.
colsToInspect=names(glob)[which(sapply(glob, class)%in% c("character","integer","numeric"))]#on recupere les noms des col character, integer et numeric
colsBoolean=names(glob)[which(sapply(glob, class)%in% c("logical"))]
colsnames = vector()
cleanGlob = data.table()
for(col in colsToInspect)
{
  if(length(unique(glob[[col]]))>2)
    {
    cleanGlob<-as.data.table(cbind(cleanGlob,glob[[col]]))
    colsnames <- c(colsnames,col)
  }
}
for(col in colsBoolean)
{
  cleanGlob<-as.data.table(cbind(cleanGlob,glob[[col]]))
  colsnames <- c(colsnames,col)
}
setnames(cleanGlob,colsnames)#on renomme le nom des colonnes

#Pour la suite de notre analyse on convertit les input (hits, pageviews) et l'output (transactionRevenue) en numeric
cleanGlob$transactionRevenue=as.numeric(cleanGlob$transactionRevenue)
cleanGlob$hits=as.numeric(cleanGlob$hits)
cleanGlob$pageviews=as.numeric(cleanGlob$pageviews)

#on verifie quelles colonnes comportent encore des NA
gg_miss_var(cleanGlob)

#Pour la colonne transactionRevenue, on va remplacer les valeurs NA par 0. En effet il y a aucune ligne
#avec une transaction revenue a 0, on peut donc supposer que ces valeurs correspond a 0.
cleanGlob$transactionRevenue[1:20]
set(cleanGlob, i = which(is.na(cleanGlob[["transactionRevenue"]])), j = "transactionRevenue", value = 0)#glob[i,transactionRevenue]=0
cleanGlob$transactionRevenue[1:20]

#Pour la colonne Pageviews on remplace les valeurs NA par la median pour ne pas influencer notre future modele
cleanGlob$pageviews[1:20]
set(cleanGlob, i = which(is.na(cleanGlob[["pageviews"]])), j = "pageviews", value = median(cleanGlob[["pageviews"]],na.rm = TRUE))#glob[i,transactionRevenue]=median(glob[,pageviews])
cleanGlob$pageviews[1:20]

#Pour la colonne isTrueDirect on s'apercoit que les seules valeurs possibles sont NA et TRUE. 
#Etant donne le nombre eleve de NA, on suppose que les valeurs NA correpondent a FALSE.
cleanGlob$isTrueDirect[1:20]
set(cleanGlob, i = which(is.na(cleanGlob[["isTrueDirect"]])), j = "isTrueDirect", value = FALSE)#glob[i,pageviews]=FALSE
cleanGlob$isTrueDirect[1:20]

#On on applique la function logarithme sur l'output pour avoir une distribution gaussienne.
#On utilise la fonction log1p qui ne va laisser les valeurs 0 a 0, cela nous permet d'eviter de nous
#retrouver avec des -inf dans la colonne transactionRevenue.
cleanGlob[,transactionRevenue:=log1p(transactionRevenue)]
cleanGlob$transactionRevenue[1:20]

#On remarques que dans certaines colonnes character les valeurs (none),(direct),(not set),unknown.unknown 
#et not available in demo dataset sont presentes. Ces valeurs correspondent a NA. On remplace
#donc ces valeurs par "ex.n
newChar=names(cleanGlob)[which(sapply(cleanGlob, class)%in% c("character"))]
for(col in newChar)
{
  set(cleanGlob, i = which(cleanGlob[[col]]=="(none)"), j = col, value = "ex.na")
}
for(col in newChar)
{
  set(cleanGlob, i = which(cleanGlob[[col]]=="(not set)"), j = col, value = "ex.na")
}
for(col in newChar)
{
  set(cleanGlob, i = which(cleanGlob[[col]]=="(direct)"), j = col, value = "ex.na")
}
for(col in newChar)
{
  set(cleanGlob, i = which(cleanGlob[[col]]=="unknown.unknown"), j = col, value = "ex.na")
}
for(col in newChar)
{
  set(cleanGlob, i = which(cleanGlob[[col]]=="not available in demo dataset"), j = col, value = "ex.na")
}

#Traitement des valeurs peu fréquentes pour les colonnes characteres, a part pour la colonne fullVisitorID.
#Ici on a un dataset de 10000 lignes on considera donc les valeurs peu frequentes comme celles aui apparaissent
#moins de 1,5% * 10000, soit moins de 150 fois.
char=names(cleanGlob)[which(sapply(cleanGlob, class)%in% c("character"))]#on recupere les noms des col character
char=char[-2]#on ne traite pas le colonne fullVisitorId
cleanGlobbis=copy(cleanGlob)
for (c in char) for (v in names(which(table(cleanGlob[[c]])<150))) cleanGlob[get(c)==v,(c):="Autre"]#remplace les valeurs peu frequentes par autre
#for (c in char) if(min(table(cleanGlobbis[[c]]))<150) {stock<-names(head(sort(table(cleanGlob[[c]])),2)) ;
#for (t in stock) cleanGlob[get(c)==t,(c):=paste(stock,collapse="_")]}

#------------------------Regroupement des données------------------------------

char<-names(cleanGlob)[which(sapply(cleanGlob, class)=="character")]
boolean<-names(cleanGlob)[which(sapply(cleanGlob, class)=="logical")]
char=char[-2]#On enleve fullvisitorID puisque cette colonne ne doit pas être factorisée
globFactor<-copy(cleanGlob)

#Certaines colonnes contiennent plus de 53 valeurs differentes, or Random.forest ne supporte pas les colonnes qui ont plus de 53  valeurs differentes.
#On doit donc diminuer le nombre de valeurs de ces colonnes. Pour ca on va supprimer les lignes qui contiennent les valeurs les plus rares de ces colonnes
#jusqu'a obtenir 53 valeurs differentes.
for (col in char)
{
  if(length(unique(globFactor[[col]]))>53)
  {
    values = names(sort(table(globFactor[[col]]),decreasing = TRUE)[54:length(table(globFactor[[col]]))])
    for (val in values)
    {
      globFactor = globFactor[!(globFactor[[col]]==val),]
    }
  }
}

#On factorise les colonnes non numeric pour randomforest et gbm 
for (c in char) globFactor[,(c):=as.factor(get(c))]
for (c in boolean) globFactor[,(c):=as.factor(get(c))]
str(globFactor)

visitors = unique(cleanGlob$fullVisitorId)#definition de la list de tous les id existants dans le dataset.
datalistRF = list()#liste de data.table, chaque data.table correspond a une ou plusieurs session d'un visiteur.
idList = 1

for(id in visitors)
{
  datalistRF[[idList]] <- globFactor[globFactor$fullVisitorId==id]

  idList = idList + 1
}
datalistRF[1:3]

#Suppression des ressources inutiles
rm(c,values,char,char1,col,colsBoolean,colsnames,colsToInspect,id,idList,j,json,newChar,v,result,temp,glob,cleanGlobbis)
gc()

#------------------------Construction des modeles------------------------------

# 1) ####### Arbre de decision #######

#Separation du training set et du testing set
set.seed(123)
split = sample.split(visitors, SplitRatio = 0.75)
train_set = subset(datalistRF, split == TRUE)
test_set = subset(datalistRF, split == FALSE)

#on transforme les listes de data.table en data.table
train_set = bind_rows(train_set)
test_set = bind_rows(test_set)

#On conserve les ID pour la suite avant de les supprimer
train_ID = train_set$fullVisitorId
test_ID = test_set$fullVisitorId

#On construit deux datatables pour stocker les resultats finaux (Output Test et Output predit)
#Chaque ligne de ces datatables correspond a un user. la premiere colonne contient l'ID de l'user.
#La deuxieme colonne correspond au log de la somme des transaction d'un user.
resultsTest = as.data.table(unique(test_set$fullVisitorId))
setnames(resultsTest,"fullVisitorId")
resultsTest[,final_transactionRevenue:=0]
resultsPred = copy(resultsTest)

#On supprime la colonne fullVisitorId qui ne sera pas utile pour l'apprentissage et l'evaluation de notre modele.
train_set$fullVisitorId <- NULL
test_set$fullVisitorId <- NULL

#Construction de l'arbre
arbre<-rpart(transactionRevenue~.,train_set)
rpart.plot(arbre,cex=0.5)#affichage de l'arbre

#Prediction sur le jeu de test
predArbre<-predict(arbre,test_set)

#On fait la somme des transactions pour chaque user. Ici on utlise la fonction exp puisque 
#la donnée à été passée par la fonction log auparavant : Y = sum(exp(yi))
for(i in 1:length(predArbre))
{
  resultsPred[fullVisitorId==test_ID[i],"final_transactionRevenue"] = resultsPred[fullVisitorId==test_ID[i],"final_transactionRevenue"] + exp(predArbre[i])
  resultsTest[fullVisitorId==test_ID[i],"final_transactionRevenue"] = resultsTest[fullVisitorId==test_ID[i],"final_transactionRevenue"] + exp(test_set$transactionRevenue[i])
}

#On fait le log de la somme auquel on ajout 1 : Y = log(Y + 1)
for(i in 1:length(resultsPred$fullVisitorId))
{
  resultsPred[i,"final_transactionRevenue"] = log(resultsPred[i,"final_transactionRevenue"] + 1)
  resultsTest[i,"final_transactionRevenue"] = log(resultsTest[i,"final_transactionRevenue"] + 1)
}
resultsPred[1:20,]
resultsTest[1:20,]

#Evaluation du modele
MSE = mse(resultsPred$final_transactionRevenue,resultsTest$final_transactionRevenue)
MSE

#Inportance des features
require(ggthemes)
impArbre<-arbre$variable.importance
impArbre=data.table(variable=names(impArbre),importance=impArbre)
impArbre[,variable:=factor(variable,levels=impArbre[order(importance)]$variable)]
ggplot(impArbre,aes(x=variable,y=importance))+geom_bar(stat="identity")+theme_tufte()+coord_flip()

#2) ####### Random Forest ########

#Separation du training set et du testing set
set.seed(123)
split = sample.split(visitors, SplitRatio = 0.75)
train_setRF = subset(datalistRF, split == TRUE)
test_setRF = subset(datalistRF, split == FALSE)

#on transforme les listes de data.table en data.table
train_setRF = bind_rows(train_setRF)
test_setRF = bind_rows(test_setRF)

#On conserve les ID pour la suite avant de les supprimer
train_IDRF = train_setRF$fullVisitorId
test_IDRF = test_setRF$fullVisitorId

#On construit deux datatables pour stocker les resultats finaux (Output Test et Output predit)
#Chaque ligne de ces datatables correspond a un user. la premiere colonne contient l'ID de l'user.
#La deuxieme colonne correspond au log de la somme des transaction d'un user.
resultsTestRF = as.data.table(unique(test_setRF$fullVisitorId))
setnames(resultsTestRF,"fullVisitorId")
resultsTestRF[,final_transactionRevenue:=0]
resultsPredRF = copy(resultsTestRF)

#On supprime la colonne fullVisitorId qui ne sera pas utile pour l'apprentissage et l'evaluation de notre modele.
train_setRF$fullVisitorId <- NULL
test_setRF$fullVisitorId <- NULL

#Construction du modele Random Forest
rF<-randomForest(transactionRevenue~.,data=train_setRF,importance=TRUE)
rF

#Prediction sur le jeu de test
predRF<-predict(rF,test_setRF)

#On fait la somme des transactions pour chaque user. Ici on utlise la fonction exp puisque 
#la donnée à été passée par la fonction log auparavant : Y = sum(exp(yi))
for(i in 1:length(predRF))
{
  resultsPredRF[fullVisitorId==test_IDRF[i],"final_transactionRevenue"] = resultsPredRF[fullVisitorId==test_IDRF[i],"final_transactionRevenue"] + exp(predRF[i])
  resultsTestRF[fullVisitorId==test_IDRF[i],"final_transactionRevenue"] = resultsTestRF[fullVisitorId==test_IDRF[i],"final_transactionRevenue"] + exp(test_setRF$transactionRevenue[i])
}

#On fait le log de la somme auquel on ajout 1 : Y = log(Y + 1)
for(i in 1:length(resultsPredRF$fullVisitorId))
{
  resultsPredRF[i,"final_transactionRevenue"] = log(resultsPredRF[i,"final_transactionRevenue"] + 1)
  resultsTestRF[i,"final_transactionRevenue"] = log(resultsTestRF[i,"final_transactionRevenue"] + 1)
}
resultsPredRF[1:20,]
resultsTestRF[1:20,]

#Evaluation du modele
MSERF = mse(resultsPredRF$final_transactionRevenue,resultsTestRF$final_transactionRevenue)
MSERF

#Importance des features
importancesRF = as.data.frame(rF$importance)
if(colnames(importancesRF)[1]=="%IncMSE"){setnames(importancesRF,"%IncMSE","IncMSE")}
ggplot(importancesRF,aes(x=IncMSE,y=IncNodePurity,label=rownames(importancesRF))) + geom_point() + geom_text(aes(label=rownames(importancesRF)),hjust=0, vjust=0)

#3) ####### GBM ########

#Separation du training set et du testing set
set.seed(123)
split = sample.split(visitors, SplitRatio = 0.75)
train_setGBM = subset(datalistRF, split == TRUE)
test_setGBM = subset(datalistRF, split == FALSE)

#on transforme les listes de data.table en data.table
train_setGBM = bind_rows(train_setGBM)
test_setGBM = bind_rows(test_setGBM)

#On conserve les ID pour la suite avant de les supprimer
train_IDGBM = train_setGBM$fullVisitorId
test_IDGBM = test_setGBM$fullVisitorId

#On construit deux datatables pour stocker les resultats finaux (Output Test et Output predit)
#Chaque ligne de ces datatables correspond a un user. la premiere colonne contient l'ID de l'user.
#La deuxieme colonne correspond au log de la somme des transaction d'un user.
resultsTestGBM = as.data.table(unique(test_setGBM$fullVisitorId))
setnames(resultsTestGBM,"fullVisitorId")
resultsTestGBM[,final_transactionRevenue:=0]
resultsPredGBM = copy(resultsTestGBM)

#On supprime la colonne fullVisitorId qui ne sera pas utile pour l'apprentissage et l'evaluation de notre modele.
train_setGBM$fullVisitorId <- NULL
test_setGBM$fullVisitorId <- NULL

#Construction du modele GBM
mod.gbm <- gbm(transactionRevenue~., distribution = "gaussian",data = train_setRF, n.trees = 20,interaction.depth = 3, n.minobsinnode = 1, shrinkage = 0.5,bag.fraction = 0.7, cv.folds = 3,train.fraction = 0.7)
mod.gbm

#Prediction sur le jeu de test
predgbm<-predict.gbm(mod.gbm,n.trees=gbm.perf(mod.gbm),test_setGBM)#gbm.perf(mod.gbm) donne le nb optimal de trees a utilisé pour la prediction  

#On fait la somme des transactions pour chaque user. Ici on utlise la fonction exp puisque 
#la donnée à été passée par la fonction log auparavant : Y = sum(exp(yi))
for(i in 1:length(predgbm))
{
  resultsPredGBM[fullVisitorId==test_IDGBM[i],"final_transactionRevenue"] = resultsPredGBM[fullVisitorId==test_IDGBM[i],"final_transactionRevenue"] + exp(predgbm[i])
  resultsTestGBM[fullVisitorId==test_IDGBM[i],"final_transactionRevenue"] = resultsTestGBM[fullVisitorId==test_IDGBM[i],"final_transactionRevenue"] + exp(test_setGBM$transactionRevenue[i])
}

#On fait le log de la somme auquel on ajout 1 : Y = log(Y + 1)
for(i in 1:length(resultsPredGBM$fullVisitorId))
{
  resultsPredGBM[i,"final_transactionRevenue"] = log(resultsPredGBM[i,"final_transactionRevenue"] + 1)
  resultsTestGBM[i,"final_transactionRevenue"] = log(resultsTestGBM[i,"final_transactionRevenue"] + 1)
}
resultsPredGBM[1:20,]
resultsTestGBM[1:20,]

#Evaluation du modele
MSEgbm = mse(resultsPredGBM$final_transactionRevenue,resultsTestGBM$final_transactionRevenue)
MSEgbm

#Importance des features
summary(mod.gbm)
