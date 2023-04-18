&rlm;


[link_Tensors]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Tensors.ipynb
[link_Neural_Networks]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Neural_Networks.ipynb
[link_Autograd_and_Backpropagation]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Autograd_and_Backpropagation.ipynb

[link_Models]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Models.ipynb
[link_Convolution_Layer]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Convolution_Layer.ipynb
[link_Embedding_Layer]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Embedding_Layer.ipynb
[link_Optimizers]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Optimizers.ipynb
[link_Loss_Functions]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Loss_Functions.ipynb
[link_Language_models]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Language_models.ipynb
[link_Images_models]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Images_models.ipynb
[link_Gan_Models]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Gan_Models.ipynb!!!!!!
[link_Hyperparameters]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Hyperparameters.ipynb !!!!
[link_Visualztion]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Visualztion.ipynb
[link_Data_prosessing]: https://colab.research.google.com/github/avrymi-asraf/NN-in-pythorch-HB/blob/main/Data_prosessing.ipynb

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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Tensors)

- [Neural_Networks](Neural_Networks.ipynb)מסביר את המבנה הבסיסי של רשתות נויירונים.➕ 
- [Autograd_and_Backpropagation](Autograd_and_Backpropagation.ipynb) מסביר איך עובד תהליך הפיעפוע לאחור  - תהליך האימון ברשתות נויירונים ואיך ב_pytorch_ התהליך הזה קורה באופן אטומטי. ➕
### מדריכים מפורטים
- [Models](Models.ipynb)  מודלים זה לב הרשת נויירונים, נראה את הממשק של pytorch  לבניית מודלים➕
- [Dence_Layer](Dence_Layer.ipynb) השיכבה הבסיסית, שכיבה ליניארית פלוס פונקציית אקטיבציה ➕
- [Convolution_Layer](Convolution_Layer.ipynb)  שכבת קונבולוציה - היא משמשת בעיקר לרשתות העוסקות בתמונות ➕
- [Embedding_Layer](Embedding_Layer.ipynb)  שכבת הטמעה - מאפשרת להציג מילים בתור וקטורים במיימדים יותר גובהים➕
- [Optimizers](Optimizers.ipynb) סוגי האופטימייזרים השונים הקיימים, והשוואות בניהם. ➕
- [Loss_Functions](Loss_functions.ipynb) פונקציית העלות השונות וההבדלים בניהם. ➕
- [Language_models](Language_models.ipynb) כלים שימושיים למודלי שפה.🔲
- [Images_models](Images_models.ipynb) כלים שימושיים למודלים העוסקים בתמונות.🔲
- [Gan_Models](Gan_Models.ipynb) יצירת מודלים יצירתיים. ➕
- [Hyperparameters](Hyperparameters.ipynb) קביעת ההיפר פרמטרים במהלך אימון הרשת.🔲 

### שונות
- [Visualztion](Visualztion.ipynb) כלים וטכינקות להמחשה של תהליך בניית הרשת ואימונה, תוך שימוש בסיפריית `plotly`. ➕
- [Data_prosessing](Data_prosessing.ipynb) תהליך יבוא וניקוי דאטא🔲
- [Tools](Tools.py) כלים שונים. 
- [Guide](Guide.md) איך לתרום למדריכים. 
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Neural_Network)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Autograd_and_Backpropagation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Models)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Convolution_Layer)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Embedding_Layer)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Optimizers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Loss_Functions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Language_models)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Images_models)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Gan_Models)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Hyperparameters)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Visualztion)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link_Data_prosessing)
