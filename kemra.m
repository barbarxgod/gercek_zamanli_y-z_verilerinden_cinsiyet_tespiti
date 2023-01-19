%201713709035 Mustafa Koray Memiş
%Programı çalıştırmadan önce gerekli olan eklentileri "kurulması gereken
%eklentiler" klasöründen yükleyiniz.
%Hızlı sonuç almak için eğitim scriptimizdeki 150. Satırdaki max epoch
%değerini daha düşük seviyelere çekebilirsiniz. Fakat en doğru sonucu
%datasetteki tüm görselleri kullanarak alabilirsiniz.

web=webcam();
%Bu kod parçacığında, webcam fonksiyonu ile bir web kamerasının çalıştırılması ve bu web kamerasının çalıştırılması için bir nesne oluşturulmuştur. 
%Bu nesneyi, web kamerasından görüntüleri almak ve bu görüntüler üzerinde işlem yapmak için kullanacağız.
%Eğer webcam kısmında hata alıyorsanız. MMatlabın webcami görebilmesi için
%gerekli olan eklenti kurulması gereken eklentiler klasöründe
%bulunmaktadır.



yuz =vision.CascadeObjectDetector();
%"vision.CascadeObjectDetector" fonksiyonu ile bir yüz tespit edici model oluşturuyoruz. 
%Bu modeli daha sonra görüntüler üzerinde kullanılarak yüzlerin tespiti için kullanacağız.
%vision.CascadeObjectDetector fonksiyonu, MATLAB'de yer alan "Computer Vision Toolbox" eklentisi ile birlikte kullanılır. 
%Bu fonksiyon, yüz tespiti için kullanılan "cascade object detection" yöntemini kullanarak bir yüz tespit edici model oluşturur. 
%Bu yöntem, yüzleri tespit etmek için çok sayıda özelik ve kural tanımlar ve bu özelikler ve kuralları kullanarak bir yüz tespit edici model oluşturur.
%yuz değişkenine atanan bu yüz tespit edici modeli daha sonra görüntüler üzerinde kullanıp yüzlerin tespiti için kullanacağız.



while true
        
    goruntu =snapshot(web);
    gri = rgb2gray(goruntu);
    bbox = step(yuz,gri);
    %Bu kod bloğunda, öncelikle "web" nesnesi üzerinden anlık bir görüntü alınır ve bu görüntü "goruntu" değişkenine atanır. 
    %Daha sonra, bu görüntü gri seviyesi bir görüntüye dönüştürülür ve bu gri seviyesi görüntü "gri" değişkenine atanır.
    %Son olarak, "yuz" yüz tespit edici modeli ile gri gri seviyesi görüntü üzerinde yüz tespiti gerçekleştirilir ve tespit edilen yüzler için 
    %koordinat bilgisi "bbox" değişkenine atanır. Bu koordinat bilgisi, yüzlerin görüntü üzerinde nerede olduğunu belirtir.
    %Bu kod parçacığı ile yüz tespiti gerçekleştirilerek yüzlerin görüntü üzerinde nerede olduğu bilgisi elde edilir. 
    
    
    
    resim = imresize(goruntu, [224, 224]);
    %"imresize" fonksiyonu ile "goruntu" görüntüsü ölçeklendirilmektedir. Ölçeklendirme işlemi, görüntünün boyutunun değiştirilmesini sağlar. 
    %Bu kısımda, görüntünün boyutunu [224, 224] olarak belirledik. Bu, görüntünün yüksekliğinin ve genişliğinin 224 piksel olacağı anlamına gelir. 
    %Bu işlem sonucu oluşan görüntü "resim" değişkenine atanır.
    %"imresize" fonksiyonu, görüntü ölçeklendirme işlemi yapmak için kullanılır. Bu fonksiyon, ilk parametresi olarak ölçeklendirilecek 
    %görüntüyü ve ikinci parametresi olarak görüntünün ölçeklendirileceği boyutu alır. Bu fonksiyon, görüntü ölçeklendirme işlemini gerçekleştirerek 
    %ölçeklendirilmiş görüntüyü döndürür.
    
    
    
    [Label, Prob] = classify(net,resim);
    %Burada "classify" fonksiyonu kullanılarak net sinir ağı ile resim görüntüsü üzerinde sınıflandırma işlemi gerçekleştirilmiştir. 
    %Bu işlem sonucu oluşan sınıf etiketi ve olasılık değerleri "Label" ve "Prob" değişkenlerine atanır.
    %"classify" fonksiyonu, bir sinir ağının verilen bir görüntü üzerinde sınıflandırma işlemi gerçekleştirmeye yarar. 
    %Bu fonksiyon, ilk parametresi olarak sınıflandırma işlemini gerçekleştirecek olan sinir ağını ve ikinci parametresi olarak sınıflandırma işlemine 
    %tabi tutulacak olan görüntüyü alır. Bu fonksiyon, sınıflandırma işlemini gerçekleştirerek sınıf etiketi ve olasılık değerlerini döndürür.
    %Burada, net sinir ağı ile resim görüntüsü üzerinde sınıflandırma işlemi gerçekleştirilir ve işlem sonucu oluşan 
    %sınıf etiketi ve olasılık değerleri "Label" ve "Prob" değişkenlerine atanır. Bu sınıf etiketi ve olasılık değerleri, 
    %daha sonra görüntünün hangi sınıfa ait olduğunu ve bu sınıfa ait olma olasılığını belirtir.



    isim=char(Label);
    %Burada, Label değişkeninin değeri char fonksiyonu ile dizgi (string) tipine dönüştürülmüş ve bu dizgi değeri isim değişkenine atanmıştır.
    %char fonksiyonu, verilen bir sayı değerini dizgi olarak dönüştürmeye yarar. 
    %Bu fonksiyon, bir sayı değeri alır ve bu değerin karşılık geldiği dizgi değerini döndürür. 
    %Örneğin, char(65) ifadesi "A" dizgisini döndürür, char(66) ifadesi ise "B" dizgisini döndürür.
    %Bu kod parçacığında, Label değişkeninin değeri char fonksiyonu ile dizgi olarak dönüştürülür ve bu dizgi değer isim değişkenine atanır. 
    %Bu sayede, Label değişkeninin değeri dizgi tipinde olur ve bu değer isim değişkenine atanır.



    deger=num2str(max(Prob));
    %Burada, Prob değişkeninin elemanlarından en büyük olanı max fonksiyonu ile bulunur ve bu değer num2str 
    %fonksiyonu ile sayı değerinden dizgi (string) değerine dönüştürülür. Dönüştürülen dizgi değeri de deger değişkenine atanır.
    %max fonksiyonu, verilen bir dizi içindeki en büyük elemanı bulmaya yarar. Bu fonksiyon, bir dizi alır ve dizinin elemanları 
    %arasından en büyük olanını döndürür. Örneğin, max([1 2 3]) ifadesi 3 değerini döndürür, max([-1 0 1]) ifadesi ise 1 değerini döndürür.
    %num2str fonksiyonu ise, verilen bir sayı değerini dizgi (string) değerine dönüştürmeye yarar. 
    %Bu fonksiyon, bir sayı değeri alır ve bu değerin karşılık geldiği dizgi değerini döndürür. Örneğin, num2str(3.14) ifadesi "3.14" dizgisini döndürür, 
    %num2str(2) ifadesi ise "2" dizgisini döndürür.
    %Burada, Prob değişkeninin elemanlarından en büyük olanı max fonksiyonu ile bulunur ve bu değer num2str 
    %fonksiyonu ile dizgi olarak dönüştürülür. 
    %Dönüştürülen dizgi değeri de deger değişkenine atanır. Bu sayede, Prob değişkeninin en büyük elemanı sayı değerinden dizgi 
    %değerine dönüştürülür ve bu dizgi değer "deger" değişkenine atanır.



    detpic=insertObjectAnnotation(goruntu,"rectangle",bbox,isim+" "+deger);
    %Burada, insertObjectAnnotation fonksiyonu kullanılarak goruntu görüntüsüne yüzleri tespit etme işlemini 
    %gerçekleştirdiğimiz kareleri eklenir ve bu karelerin içine de görüntülerin sınıfı ve sınıfın olasılığı bilgileri yazdırılır.
    %insertObjectAnnotation fonksiyonu, verilen bir görüntünün üzerine nesneler için etiketler (annotation) eklemeye yarar. 
    %Bu fonksiyon, görüntü, etiketleri eklenecek nesnelerin koordinatları, etiketlerin yazılacağı yerler ve etiketlerin içeriği gibi 
    %parametreler alır ve etiketleri eklenmiş görüntüyü döndürür. Örneğin, insertObjectAnnotation
    %(I, 'rectangle', [10 10 20 20], 'Etiket') ifadesi I görüntüsünün (10,10) ve (20,20) koordinatları arasında bir kare çizerek 
    %bu karenin içine "Etiket" yazısını ekleyerek dönüştürülmüş görüntüyü döndürür.
    %Burada, insertObjectAnnotation fonksiyonu ile goruntu görüntüsüne tespit edilen yüzleri gösteren kareler 
    %eklenir ve bu karelerin içine de görüntülerin sınıfı ve sınıfın olasılığı bilgileri yazdırılır. 
    %Bu bilgiler, isim ve deger değişkenlerinde saklanmaktadır. Bu sayede, yüzleri tespit eden karelerin 
    %içine görüntülerin sınıfı ve sınıfın olasılığı bilgileri eklenir ve bu bilgiler detpic değişkenine atanır.     



     imshow(detpic); 
    %Burada, imshow fonksiyonu kullanılarak detpic değişkeninde saklanan görüntü gösterilir.
    %imshow fonksiyonu, verilen bir görüntüyü ekranda göstermeye yarar. 
    %Bu fonksiyon, görüntünün saklandığı bir matris veya dosya yolu gibi parametreler alır ve görüntüyü ekranda gösterir. 
    %Örneğin, I = imread('goruntu.jpg'); imshow(I) ifadesi "goruntu.jpg" dosyasından okunan görüntüyü ekranda gösterir.
    %Bu kod parçacığında, imshow fonksiyonu ile detpic değişkeninde saklanan görüntü ekranda gösterilir. 
    %Bu sayede, yüzleri tespit eden karelerin içine görüntülerin sınıfı ve sınıfın olasılığı bilgilerinin eklenmiş hali gösterilebilir. 
    %Bu görüntünün ekranda gösterilmesi, görüntünün gerçek zamanlı olarak güncellenmesi veya değiştirilmesi gibi amaçlarla kullanılabilir.  
end
%Bu kod parçacıklarının while döngüsü içerisinde yer alması, görüntülerin gerçek zamanlı olarak işlenmesini ve güncellenmesini sağlar. 
%While döngüsü sayesinde, bu kod parçacıkları sürekli olarak tekrar edilir ve bu sayede görüntüler sürekli olarak yenilenir. 
%Böylece, görüntülerde tespit edilen yüzlerin sınıfı ve sınıfın olasılığı bilgileri de sürekli olarak güncellenir ve gösterilir.


