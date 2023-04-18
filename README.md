&rlm;


# עקרונות של רשתות  ע"י _pytorch_

מדריך בעברית לשימוש בספרייה 
_pytorch_


## מבנה
&rlm;
המדריך מחולק ל2 חלקים: החלק הבסיסי מסביר את העקרונות הבסיסים שאותם צריך בשביל לבנות רשת נויירונים. 
בחלק הזה נראה רק דוגמאות חלקיות, כיון שהמטרה היא רק להבין איך קורה תהליך יצירה ואימון של רשת נויירונים. 
בחלק המתקדם יש פירוט על כל רכיב ורכיב בפני עצמו. ההמלצה שלי היא לעבור על החלק הבסיסי, ורק לאחר מכן להעמיק לפי הצורך בחלקים השונים. 

ריק 🔲
דרוש תוספת תוכן ➕
דרוש עריכה📝
### מדריך בסיסי
- [Tensors](Tensors.ipynb) טזנסור הוא הבסיס עליו בנוייה הספרייה.📝 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Tensors.ipynb)

- [Neural_Networks](Neural_Networks.ipynb)מסביר את המבנה הבסיסי של רשתות נויירונים.➕ 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Neural_Networks.ipynb)

- [Autograd_and_Backpropagation](Autograd_and_Backpropagation.ipynb) מסביר איך עובד תהליך הפיעפוע לאחור  - תהליך האימון ברשתות נויירונים ואיך ב_pytorch_ התהליך הזה קורה באופן אטומטי. ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Autograd_and_Backpropagation.ipynb)

### מדריכים מפורטים
- [Models](Models.ipynb)  מודלים זה לב הרשת נויירונים, נראה את הממשק של pytorch  לבניית מודלים➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Models.ipynb)

- [Linear_and_Activation_Layer](Linear_and_Activation_Layer.ipynb) 
השיכבה הבסיסית, שכיבה ליניארית פלוס פונקציית אקטיבציה ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Linear_and_Activation_Layer.ipynb)

- [Convolution_Layer](Convolution_Layer.ipynb)  שכבת קונבולוציה - היא משמשת בעיקר לרשתות העוסקות בתמונות ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Convolution_Layer.ipynb)

- [Embedding_Layer](Embedding_Layer.ipynb)  שכבת הטמעה - מאפשרת להציג מילים בתור וקטורים במיימדים יותר גובהים➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Embedding_Layer.ipynb)

- [Optimizers](Optimizers.ipynb) סוגי האופטימייזרים השונים הקיימים, והשוואות בניהם. ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Optimizers.ipynb)

- [Loss_Functions](Loss_functions.ipynb) פונקציית העלות השונות וההבדלים בניהם. ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Loss_Functions.ipynb)

- [Language_models](Language_models.ipynb) כלים שימושיים למודלי שפה.🔲
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Language_models.ipynb)

- [Images_models](Images_models.ipynb) כלים שימושיים למודלים העוסקים בתמונות.🔲
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Images_models.ipynb)

- [Generative models](Generative_models.ipynb) יצירת מודלים יצירתיים. ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Generative_models.ipynb)


- [Hyperparameters](Hyperparameters.ipynb) קביעת ההיפר פרמטרים במהלך אימון הרשת.🔲 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Hyperparameters.ipynb)


### שונות
- [Visualztion](Visualztion.ipynb) כלים וטכינקות להמחשה של תהליך בניית הרשת ואימונה, תוך שימוש בסיפריית `plotly`. ➕
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Visualztion.ipynb)

- [Data_prosessing](Data_prosessing.ipynb) תהליך יבוא וניקוי דאטא🔲
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Data_prosessing.ipynb)

- [Tools](Tools.py) כלים שונים. 
- [Guide](Guide.md) איך לתרום למדריכים. 
 


