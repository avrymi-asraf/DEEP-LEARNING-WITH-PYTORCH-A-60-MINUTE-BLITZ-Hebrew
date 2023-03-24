## Import
Delete what you don't want to import. 
Each new segment should import only the things needed for that segment. This new section is where there is an H1 highlight.

```python
import torch            
form torch import nn    #for
import torchtext        #for
import torchimage       #for

import numpy as np      #for
import pandas as pd     #for


import plotly.graph_objects as go 
import plotly.express as px
from plotly  import subplots
```

### Typing
We want to use  Typing only in functions, Every type that is not basic type like `List` or `Tuple` need explicit declaration
#Typing
```python
form typing import List,Tuple,Dict,Optional,Callable
ImageType = torch.Tensor
TensorType = torch.Tensor
LossFucntionType = Callable[[Tensor, Tensor], Tensor]
OptimazerType = Callable[[], None]
DataType = Tuple[Tensor, Tensor]
ParametersType = Tuple[List[float], float]
GetParaetersType = Callable[[], ParametersType]
PlotFunctionType = Callable[[Module, Tensor, Tensor, Any], Figure]
```


# Glossary
_embdding layer_ = 
_linear layer_ = 
_activation layer_ = 
_backpropgation_=
_neural network_ = 
_convolution layer_ = 
_loos function_ = 
_optimizer_ = 
_weights_ = 
_bios_ = 
_data set_ = 
_train set_ = 


[ ] structure of page
[ ] language
