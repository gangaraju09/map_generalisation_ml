# Map Generalization Project Using Machine Learning
The sample project is an attempt to study and analyze map generalization efficiency through application of Machine Learning (ML).

The repository contains three key folders - <b> Pregeneralized_Shapefiles, Generalized_Shapefiles, Vertices_Labels </b>. Pregeneralized_Shapefiles folder contains currently shapefiles of <i>California, Florida, Idaho, Louisiana, Maine, North Carolina, Texas</i>, the same set of shapefiles are present in Generalized_Shapefiles folder. The generalization is performed using <b>Viswalingam - Whyatt</b> (Weighted Area) technique. The shapefile of United States is downloaded from U.S Census Bureau and each state is exported via ArcGIS Pro as seperate shapefile and the generalization is performed usingh the website - <a href = "https://mapshaper.org/">mapshaper.org</a> developed by <b>Matthew Bloch</b>. All the generalizations are performed with a threshold of 0.30% for a fair comparison.
