# KNN Lending Analysis

هذا المشروع يهدف إلى تحليل بيانات القروض (Lending Club data) باستخدام خوارزمية **K-Nearest Neighbors (KNN)** لتصنيف وتوقع نتائج القروض.

## وصف المشروع
- استخدام لغة Python ومكتبة Scikit-learn لتطبيق خوارزمية KNN.
- معالجة البيانات وتحليل العلاقة بين المتغيرات المختلفة.
- تقييم أداء النموذج (Model Evaluation) واختيار أفضل عدد للجيران (n_neighbors) لتقليل الـ Overfitting.

## المتطلبات (Requirements)
للتشغيل، تحتاج إلى تثبيت المكتبات التالية:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## كيفية التشغيل
1. قم بتحميل الملف `neighbors_method.py`.
2. تأكد من وجود ملف البيانات `dataset_small.pkl` في نفس المجلد.
3. قم بتشغيل الكود باستخدام أمر:
   ```bash
   python neighbors_method.py
