DeeSOM  
--------------------------------------------------------------------------------
Self-organized map based classifier, developed to deal with large and highly imbalanced data.

sinc(i) - http://sinc.unl.edu.ar

The methods automatically build several layers of SOM. Data is clustered and samples that
are not likely to be positive class member are discarded at each level.  

The elastic-deepSOM (elasticSOM) is a deep architecture of SOM layers where the map
size is automatically set in each layer according to the data filtered in each previous 
map. The ensemble-elasticSOM (eeSOM) uses several SOMs in ensemble layers to
face the high imbalance challenges. These new models are particularly suited
to handle problems where there is a labeled class of interest (positive
class) that is significantly under-represented with respect to a higher number
of unlabeled data.

This code can be used, modified or distributed for academic purposes under GNU
GPL. Please feel free to contact with any issue, comment or suggestion.

This code was used in:

"Deep neural architectures for highly imbalanced data in bioinformatics"
L. A. Bugnon, C. Yones, D. H. Milone and G. Stegmayer*, IEEE Transactions on Neural Networks and Learning Systems,
 Special Issue on Recent Advances in Theory, Methodology and Applications of Imbalanced Learning (in press).

## Instalation

## Running the demo.