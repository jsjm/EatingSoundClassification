library(corrplot)
library(ggdendro)

# load csv file
setwd("/analysis")
getwd()
M <- read.csv('res_matrix.csv', header=FALSE)
names(M) <- c('aloe','burger','cabbage','candied_fruits','carrots','chips','chocolate','drinks','fries','grapes','gummies','ice-cream','jelly','noodles','pickles','pizza','ribs','salmon','soup','wings')
row.names(M) <- names(M)

# correlation matrix plot - visualizing pairwise classification accuracy
DF_M <- data.frame(M)
pmfY <- apply(as.matrix(DF_M),c(1,2),as.numeric)
pmfY
M <- pmfY
diag(M) = NA
#numbered
corrplot(M, is.corr = FALSE, type = "upper", method = "number",number.cex = .7)
# dots
corrplot(M, is.corr = FALSE, type = "upper")


# prepare hierarchical cluster
hc_M = hclust(as.dist(1-M))
hc = hclust(dist(M))

#unrooted dendro
plot(as.phylo(hc_M), type = "unrooted", cex = 0.6, no.margin = TRUE)

#ggdendro
ggdendrogram(hc_M, rotate = TRUE)


