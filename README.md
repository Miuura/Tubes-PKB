[ Klasifikasi Sirtuin 6 Small Molecules ]

Project  kelompok ini merupakan project akhir untuk mata kuliah Introduction to AI, di mana saya bertanggung jawab untuk merancang model KNN dan Decision Tree, serta melakukan evaluasi dan tuning parameter untuk setiap model.

[ Dataset ]

Dataset telah disediakan pada soal dan terdiri dari 100 molekul dengan 6 deskriptor untuk menentukan kandidat inhibitors protein Sirtuin 6. Data dibagi menjadi dua kelompok yaitu low BFE dan high BFE.

[ Preprocessing ]

Dalam tahap preprocessing, kami mengidentifikasi beberapa masalah dalam dataset yang dapat memengaruhi kualitas model:

- Nilai Unik: Terdapat nilai unik dalam fitur SC-6 dan maxwHBa, di mana setiap nilai dalam fitur tersebut berbeda. Fitur dengan nilai unik seperti ini dapat menyebabkan model, khususnya Decision Tree, melakukan split yang sangat spesifik dan mungkin tidak dapat di generalisasi dengan baik pada data baru.
- Banyak Nilai 0: Fitur SHBd dan minHaaCH memiliki banyak nilai 0, yang bisa disebabkan oleh outlier atau data yang tidak relevan. Untuk menangani hal ini, kami menggunakan visualisasi boxplot untuk mengidentifikasi dan menganalisis outlier. Setelah itu, kami menerapkan metode Interquartile Range (IQR) untuk menghapus atau menangani nilai outlier, guna memastikan bahwa model tidak terpengaruh oleh data yang tidak konsisten atau ekstrem.
- Kami juga membagi dataset menjadi dua pasang: satu dengan data yang discaling dan satu lagi tanpa scaling. Hal ini bertujuan untuk mengevaluasi dampak dari data scaling pada performa model, terutama untuk KNN yang sensitif terhadap skala fitur.

[ Eksperimen dan Analisis ]

- Decision Tree: Model Decision Tree menunjukkan akurasi yang sangat baik pada data training, mendekati 1.0. Namun, akurasi pada data test berkisar antara 0.8 hingga 0.85, menunjukkan adanya overfitting. Perbedaan signifikan antara akurasi data train dan test, serta nilai cross-validation yang rendah (0.73), menandakan bahwa model tidak dapat menangani variasi data dengan baik. Hal ini mungkin disebabkan oleh fitur dengan nilai unik yang menyebabkan model membentuk banyak cabang yang terlalu spesifik untuk data train.
- KNN: Model KNN, sebagai algoritma non-parametrik, menunjukkan hasil yang lebih stabil. KNN tidak membangun model eksplisit seperti Decision Tree, sehingga lebih tahan terhadap overfitting pada data dengan nilai unik. Kami membandingkan performa KNN pada data scaled dan tanpa scaling, serta melakukan tuning parameter. Hasilnya, KNN dengan data scaled memberikan akurasi training sebesar 0.82, akurasi test sebesar 0.85, dan rata-rata cross-validation score sebesar 0.8125. KNN menunjukkan stabilitas yang lebih baik dan jarak antara train score dan cross-validation score yang tidak terlalu jauh, mengindikasikan generalisasi model yang lebih baik.

[ Kesimpulann ]

Pemilihan model machine learning harus disesuaikan dengan karakteristik dataset. Pada dataset ini, KNN terbukti lebih tahan terhadap overfitting dibandingkan Decision Tree, terutama karena KNN tidak membangun model yang terlalu spesifik untuk data training. Decision Tree, meskipun memberikan akurasi tinggi pada data training, tidak dapat menggeneralisasi dengan baik pada data test karena masalah overfitting yang disebabkan oleh nilai unik dalam fitur.

Project ini mengajarkan pentingnya memahami karakteristik data dan melakukan preprocessing yang tepat. Fine-tuning parameter dan evaluasi model adalah kunci untuk menghindari overfitting dan meningkatkan performa model. Dengan melakukan eksperimen pada data scaling dan teknik penanganan outlier, kami dapat mengidentifikasi model yang paling stabil dan efektif untuk dataset dengan fitur yang memiliki nilai unik atau outlier.
