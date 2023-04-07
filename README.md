# Map Generalization Project Using Machine Learning
The sample project is an attempt to study and analyze map generalization efficiency through application of Machine Learning (ML).

The repository contains <b>three key</b> folders:
<ul>
<li> <b> Pregeneralized_Shapefiles </b> </li> <p>Folder contains shapefiles of <i>California, Florida, Idaho, Louisiana, Maine, North Carolina, Texas</i> downloaded from <a href = "https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html"> United States Census Bureau </a> Cartographic Boundary Files - Shapefile link.</p>
<li><b> Generalized_Shapefiles </b> </li> <p> Folder contains shapefiles of <i>California, Florida, Idaho, Louisiana, Maine, North Carolina, Texas.</i> Generalization is performed using <b>Viswalingam - Whyatt</b> (Weighted Area) technique. The shapefiles are seperated and exported individually using ArcGIS Pro to <a href = "https://mapshaper.org/">mapshaper.org</a> developed by <b>Matthew Bloch</b>. All the generalizations are performed with a  0.30% simplify for a fair comparison and analysis.</p>
<li><b>Vertices_Labels</b></li> <p> Folder contains CSV files of aforementioned of shapefiles in Generalized_Shapefiles. Each CSV file has columns <b> Latitude, Longitude, Case</b>. Case columns contains <i>"Yes/No"</i> based on Latitude, Longitude columns of Generalized_Shapefiles. </p>

Sample Workflow:
<img src="https://github.com/gangaraju09/map_generalisation_ml/blob/main/DesignAssests/CartoAI_Sample_WorkFlow.png" alt="CartoAI Workflow">


