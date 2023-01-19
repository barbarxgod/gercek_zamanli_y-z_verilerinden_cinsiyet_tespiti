Dataset = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%ilk olarak matlabın çalışma alanına verisetini yüklüyoruz. Ardından veri kümesini yüklemek için imageDatastore işlevini
%kullanıyoruz. bu işlevin ilk argümanı yüklemek istediğimiz veri kümesinin yoludur, veri kümemiz çalışma dizinindedir bu nedenle
%yalnızca verilerin adını yazıyoruz. Veri setimiz alt klasörler halinde düzenlenmiştir bu nedenle include kullanıyoruz
%ve bu argümanın değerini "true" olarak ekliyoruz bu veri kategorisini etiketlememiz gereken alt klasörleri dahil ettiğimiz anlamına
%gelir aslında bu klasör adları verilerimizin etiketleridir bu nedenle "labelsource" yani etiketimizin kaynağı "foldernames"
%yani dosya adlarımızdır. Şimdi verikümemize sahibiz.



[Training_Dataset, Validation_Dataset] = splitEachLabel(Dataset, 0.7);
%Bu satırda ise datasetimizi eğitim ve doğrulama veri kümelerine ayırıyoruz. Bunun için "splitEachLabel" yani
%her bir etiketi bölme işlevini kullanıyoruz. Parantez içerisinde (dataset, 0.7) dataseti hangi oranda eğitim ve
%doğrulama olarak ayıracağımızı giriyoruz. %70 eğitim ve %30 doğrulama olarak ayırdım.



net = googlenet; 
%googleneti yükleyip onu eğitecek olduğumuz net değişkenine atıyoruz.
%googlenet kullanabilmek için gerekli olan eklenti kurulması gereken
%eklentiler klasöründe bulunmaktadır.


analyzeNetwork(net);
%googlenet mimarisini görebilmek için "analyzeNetwork" fonksiyonunu
%kullanıyoruz. Bu fonksiyonun argümanı az önce tanımladığımız "net"
%değişkenidir.



Input_Layer_Size = net.Layers(1).InputSize;
%googlenet giriş katmanı ve boyutunu "Input_Layer_Size adlı değişkenimizde
%tutuyoruz.



Layer_Graph = layerGraph(net); 
%"layerGraph" fonksiyonu ile "net" nesnesinden bir katman grafiği oluşturuyoruz.



Feature_Learner = net.Layers(142);   
Output_Classifier = net.Layers(144);
%Feature_Learner ve Output_Classifier değişkenlerini GoogLeNet ağındaki belirli katmanlara atıyoruz.
%"net.Layers" özelliği, ağdaki katmanların bir katman dizisini döndürür, bu dizideki her bir eleman ağdaki bir katmanı temsil eder 
%ve elemanın indisi katmanın ağda bulunduğu pozisyonu temsil eder.
%Feature_Learner değişkeni, ağdaki 142. katmana atanır ve Output_Classifier değişkeni, ağdaki 144. katmana atanır. 



Number_of_Classes = numel(categories(Training_Dataset.Labels));
%Eğitim verisetindeki etiketlerin kategorilerinin sayısını bulur. 
%"Training_Dataset.Labels" ifadesi, eğitim verisetindeki etiketlerin bir dizisini döndürür ve "categories" fonksiyonu bu dizinin kategorilerine böler. 
%"numel" fonksiyonu ise, bir dizinin eleman sayısını döndürür. Bu yüzden, "Number_of_Classes" değişkeni, 
%eğitim verisetindeki etiketlerin kategorilerinin sayısını temsil eder.



New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Uygun katman', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
New_Classifier_Layer = classificationLayer('Name', 'Son katman');
%Bu kod bloğu, yeni bir tam bağlı katman ve yeni bir sınıflandırma katmanı oluşturur.
%"fullyConnectedLayer" fonksiyonu, bir tam bağlı katman oluşturmak için kullanılır. 
%Bu katman, girişler ile ağırlıklar arasında çarpım yaparak ve bias ekleyerek çıkış üretebilir. 
%Bu fonksiyon, ilk parametresi olarak verilen çıkış sayısını alır ve diğer parametrelerle katmanın özelliklerini belirler. 
%'Name' parametresi katmanın adını belirler, 'WeightLearnRateFactor' ve 'BiasLearnRateFactor' parametreleri ise ağırlık ve bias öğrenme oranını belirler. 
%"Number_of_Classes" değişkenini katmanın çıkış sayısı olarak veriyoruz ve katmanın adını "Uygun katman" olarak belirliyoruz.
%"classificationLayer" fonksiyonu ise, bir sınıflandırma katmanı oluşturmak için kullanılır. Bu katman, girişlerine göre bir sınıf etiketi verebilir. 
%Bu fonksiyon, 'Name' parametresi ile katmanın adını belirleyebilir. Ben sınıflandırma katmanının adını "Son katman" olarak belirledim.



Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
%"replaceLayer" fonksiyonu, bir katman grafiğinde bir katmanı değiştirme işlemini yapar. 
%Bu fonksiyon, ilk parametresi olarak bir katman grafiği, 
%ikinci parametresi olarak değiştirilecek olan katmanın adı ve üçüncü parametresi olarak yeni katmanı alır. 
%"Layer_Graph" değişkenini katman grafiği olarak veriyoruz. ve Feature_Learner.Name
%ifadesi değiştirilecek olan katmanın adını, New_Feature_Learner değişkeni ise yeni katmanı temsil eder. 
%Bu şekilde, Layer_Graph değişkenindeki katman grafiğinde Feature_Learner katmanı New_Feature_Learner katmanı ile değiştirilmiş olur.



Layer_Graph = replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);
analyzeNetwork(Layer_Graph)
%"replaceLayer" fonksiyonu, bir katman grafiğinde bir katmanı değiştirme işlemini yapar. Bu fonksiyon, ilk parametresi olarak bir katman grafiği, 
%ikinci parametresi olarak değiştirilecek olan katmanın adı ve üçüncü parametresi olarak yeni katmanı alır. 
%"Layer_Graph" değişkenini katman grafiği olarak veriyoruz ve "Output_Classifier.Name" ifadesi değiştirilecek olan katmanın adını, 
%"New_Classifier_Layer" değişkeni ise yeni katmanı temsil eder. 
%Bu şekilde, "Layer_Graph" değişkenindeki katman grafiğinde "Output_Classifier" katmanı "New_Classifier_Layer" katmanı ile değiştirilmiş olur.
%"analyzeNetwork" fonksiyonu ise, bir ağın mimarisini inceler. Bu fonksiyon, bir ağın katmanlarını, 
%ağırlıklarını, bağlantılarını ve diğer özelliklerini görselleştirir. 
%"analyzeNetwork" fonksiyonu "Layer_Graph" değişkenindeki katman grafiğini inceler ve ağın mimarisini görselleştirir. 
%Bu sayede, ağın yeni hali hakkında bilgi edinip, ağın performansını değerlendirebiliriz.



Pixel_Range = [-30 30];
Scale_Range = [0.9 1.1];
%"Pixel_Range" dizisi, görüntüler üzerinde yapılacak olan y-ekseni ve x-ekseni çapraz çevirme işlemleri için gereken değerleri tutar. 
%Bu dizideki ilk eleman, görüntülerin y-eksenine ne kadar kaydırılacağını, ikinci eleman ise görüntülerin x-eksenine ne kadar kaydırılacağını belirtir. 
%Ben burada görüntülerin y-eksenine ve x-eksenine 30 piksel kadar kaydırılmasını istedim.

%"Scale_Range" dizisi ise, görüntüler üzerinde yapılacak olan y-ekseni ve x-ekseni ölçeği değiştirme işlemleri için gereken değerleri tutar. 
%Bu dizideki ilk eleman, görüntülerin y-eksenine ne kadar ölçeklendirileceğini, 
%ikinci eleman ise görüntülerin x-eksenine ne kadar ölçeklendirileceğini belirtir. 
%Ben burada görüntülerin y-eksenine ve x-eksenine 0.9 ile 1.1 arasında ölçeklendirilmesini istedim. 
%Bu dönüştürmeler, ağın performansını arttırmak için yapılır ve ağın daha iyi generalizasyon sağlamasını amaçlar.



Image_Augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', Pixel_Range, ...
    'RandYTranslation', Pixel_Range,... 
     'RandXScale', Scale_Range, ...
     'RandYScale', Scale_Range);
%"imageDataAugmenter" fonksiyonu, bir görüntü veriseti üzerinde dönüştürmeler yapmak için kullanılır. 
%Bu fonksiyon, birçok parametre alarak görüntüler üzerinde ne tür dönüştürmelerin yapılacağını belirler. 
%Örneğin, 'RandXReflection' parametresi, görüntülerin y-eksenine göre yansıtılıp yansıtılmayacağını belirtir. 
%'RandXTranslation' ve 'RandYTranslation' parametreleri ise görüntülerin x-eksenine ve y-eksenine ne kadar kaydırılacağını belirtir. 
%'RandXScale' ve 'RandYScale' parametreleri ise görüntülerin x-eksenine ve y-eksenine ne kadar ölçeklendirileceğini belirtir. 
%Burada yukarda tanımlamış olduğumuz "Pixel_Range" ve "Scale_Range" dizileri bu parametreler için değerler olarak verilmiştir. 
%Bu sayede, Image_Augmenter nesnesi görüntüler üzerinde yansıma, kaydırma ve ölçekleme gibi dönüştürmeleri yapabilecektir.
%Bu "imageDataAugmenter" nesnesi, daha sonra eğitim verisetinde kullanılacak ve ağın performansını arttırmak için görüntülerin çeşitliliğini 
%artırmak amacıyla kullanılacaktır. Bu sayede, ağ daha çeşitli görüntüler üzerinde eğitilerek daha iyi generalizasyon sağlayabilecektir.
%Bu nesne, dönüştürmeleri gerçekleştirirken rastgele değerler seçerek dönüştürmeleri uygular. 
%Bu sayede, aynı görüntü üzerinde aynı dönüştürmeyi birden fazla kez yapmayacak ve dönüştürmeler arasında çeşitlilik sağlanacaktır. 
%Bu da ağın daha iyi generalizasyon sağlamasını amaçlamaktadır.  
 
 

Augmented_Training_Image = augmentedImageDatastore(Input_Layer_Size(1:2), Training_Dataset, ...
    'DataAugmentation', Image_Augmenter);
Augmented_Validation_Image = augmentedImageDatastore(Input_Layer_Size(1:2),Validation_Dataset);
%"augmentedImageDatastore" fonksiyonu, bir görüntü verisetini dönüştürmeler uygulanacak şekilde yeniden yapılandırır. 
%Bu fonksiyon, ilk parametresi olarak görüntülerin boyutlarını, ikinci parametresi olarak dönüştürmeler uygulanacak olan verisetini ve üçüncü parametresi 
%olarak dönüştürmeleri gerçekleştirecek olan "imageDataAugmenter" nesnesini alır. Kodun bu kısmında, "Input_Layer_Size(1:2)" ifadesi ağın giriş katmanının 
%boyutlarını, "Training_Dataset" değişkeni ise eğitim verisetini ve "Image_Augmenter" nesnesi ise dönüştürmeleri gerçekleştirecek olan nesneyi temsil eder. 
%Bu şekilde, "Augmented_Training_Image" değişkeni eğitim verisetini dönüştürmeler uygulanacak şekilde yeniden yapılandırır ve bu değişken 
%eğitim için kullanılacak olan verisetini temsil eder. Aynı şekilde, "Augmented_Validation_Image" değişkeni de doğrulama verisetini dönüştürmeler 
%uygulanacak şekilde yeniden yapılandırır ve bu değişken doğrulama için kullanılacak olan verisetini temsil eder.



Size_of_Minibatch = 64;
Validation_Frequency = floor(numel(Augmented_Training_Image.Files)/Size_of_Minibatch);
Training_Options = trainingOptions('sgdm',... 
    'MiniBatchSize', Size_of_Minibatch, ...
    'MaxEpochs', 100,...
    'InitialLearnRate', 3e-4,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Augmented_Validation_Image, ...
    'ValidationFrequency', Validation_Frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
%Bu kod bloğunda, ağın eğitim seçeneklerini belirleyen bir "trainingOptions" nesnesi oluşturuyoruz.
%Bu "trainingOptions" nesnesi, daha sonra "trainNetwork" fonksiyonu ile ağın eğitileceği zaman kullanılacaktır.
%Bu kod bloğunda, "Size_of_Minibatch" değişkeni ağın kaç adet görüntü üzerinde eğitileceğini, 
%"Validation_Frequency" değişkeni ise ağın kaç kere doğrulama veriseti üzerinde doğrulama yapılacağını belirtir. 
%Bu değişkenler, "trainingOptions" nesnesine 'MiniBatchSize' ve 'ValidationFrequency' parametreleri ile verilir. 
%Ayrıca, bu kod bloğunda ağın eğitileceği optimizasyon algoritması olarak 'sgdm', ağın kaç kere eğitileceği olarak "100" ve ağın eğitim sırasında 
%kullanacağı öğrenme oranı olarak "3e-4" değerleri verilmiştir. Bu seçenekler ile birlikte, 'Shuffle' parametresi ile verisetinin eğitim sırasında 
%karıştırılıp karıştırılmayacağı belirtilmiştir.
%'ValidationData' parametresi, ağın eğitim sırasında kullanacağı doğrulama verisetini temsil eder. 
%'ValidationFrequency' parametresi ise ağın kaç kere doğrulama veriseti üzerinde doğrulama yapılacağını belirtir. 
%'Verbose' parametresi ile ağın eğitim sırasında gösterilecek olan eğitim ilerlemesinin gösterilip gösterilmeyeceği belirtilir. 
%'Plots' parametresi ile ise ağın eğitim sırasında gösterilecek olan eğitim ilerlemesinin hangi türde olacağı belirtilir.



net = trainNetwork(Augmented_Training_Image, Layer_Graph, Training_Options);
%Bu kod parçacığında, "trainNetwork" fonksiyonu ile ağ veriseti üzerinde eğitilerek öğrenme gerçekleştirilir. 
%Bu fonksiyon, ilk parametresi olarak eğitim verisetini, ikinci parametresi olarak ağ modelini ve üçüncü parametresi olarak eğitim seçeneklerini 
%belirleyen "trainingOptions" nesnesini alır. Bu şekilde, ağ veriseti üzerinde belirtilen seçenekler doğrultusunda eğitilerek öğrenme gerçekleştirilecektir.



%****************************************
%Confusion chart kısmı

% Test verisetinde tahminleri yap
predictions = classify(net, Augmented_Validation_Image);
% Test verisetindeki gerçek etiketleri al
labels = Validation_Dataset.Labels;
% Confusion chart'ını çiz
figure
confusion_chart = confusionchart(labels, predictions);
%****************************************

%****************************************
%Confusion matrix kısmı
% Test verisetinde tahminleri yap
predictions = classify(net, Augmented_Validation_Image);

% Test verisetindeki gerçek etiketleri al
trueLabels = Validation_Dataset.Labels;

% Confusion matrix'ini çiz
figure
plotconfusion(trueLabels, predictions)
%****************************************





