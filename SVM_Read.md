# SVM模型最優解參數挑選
首先使用pandas套件引入資料，同時drop掉對TOP100預測沒有幫助的公司代碼以及名稱，並分離特徵值以及目標值，接下來便是一系列對於SVM的參數挑選，嘗試找出區域的最佳參數解。

## Step 1.kernel函數挑選
由於SVM可以替換掉核心的kernel核函數以達到**投射特徵**的效果，因此最一開始的便是要測試不同的透射函數對於預測的效果哪個較佳，
而後才能在較佳的kernel函數上做最佳參數的挑選。
在這個部份使用四個比較主流的kernel核函數，其中linear為簡單線性核函數，poly為可以實現更高次方的多項式核函數，rbf為輻射基底核函數，又被稱為高斯核函數，以及最後一個sigmoid乙狀核函數
，其中除了linear為簡單線性以外其他三種核函數都能做到非線性的界線劃分，以下為測試四個不同核函數的大致表現：
<div align=center><img width="650" height="400" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/29c787c7-e45d-4baf-babc-c4525cf13440"/></div>
可以看到說linear以及rbf的表現明顯較佳，因此下一步為針對這兩種核函數做模型參數的最優解測試，嘗試找出較佳的參數，看有沒有辦法再進一步的去增加模型的效能。

## Step 2.參數C及gamma選擇
C為SVM中一個會大幅影響模型表現的參數，是一種**懲罰項Regularization term**，也就是一種用來**調整模型權重的參數**，由於SVM在劃定不同分類的界線時有可能會出現很多種不同的劃分方式，
有時候為了完全劃清不同類別的資料會出現overfitting的狀況，有時候有可能會出現劃分不清的狀況，而C的調整就可以在這部分上調整SVM模型在資料劃分上的完全性或通用性。

若我們選擇了一個**較大的C**，則此時模型對於分類錯誤的資料會有較大的懲罰項，導致模型會趨向於**完全劃分**所有資料點的情形，因此有時候C取得太大反而會有**overfitting**的狀況；

而如果選擇了一個**較小的C**，則這時候因為即便分類錯誤的話對模型也不會有太大的懲罰項影響，因此可能會造成說模型的**分類效果不太佳**。

因此在選擇C上需要搭配一個test做測試，由於選擇C以及模型訓練的過程中不會接觸到test資料集，因此就可以用test資料集的表現去觀察哪一個C可以在test上面有較好的表現，
而不是單純只在train資料集上面有較佳的表現。

----------------------------------------------------------------------------------------------------

Gamma函數為**rbf核函數**中特有的參數，決定了資料**投射到高維度空間後的曲率**，因此簡單線性核函數linear**並不具有**gamma參數可以調整。

當gamma較小時，rbf核函數的曲線**較為平緩**，影響範圍較廣，也較為**全局性**，它會使劃分的界線較平滑，讓模型不容易overfitting，但若值太小導致沒辦法捕捉到一些居部的特性反而會導致underfitting；

而當gamma較大時，rbf核函數的曲線**較為陡峭**，影響範圍較窄，較為**局部性**，可以讓模型去捕捉局部的變化，然而帶來的就是有可能會overfitting的風險，以及犧牲掉了全局的特性。

### Step 2-1.針對linear核函數的參數C選擇
以下為針對linear核函數在C上的選擇，其中以分離出來的test資料集表現為縱軸，可以看出由於linear模型較簡單，因此較不容易出現overfitting的狀況，在C大於3的狀況test的表現都不會下降，考慮到linear模型較簡單的狀況，因此這邊選擇C=10來當作後續測試時C的參數選擇。
<div align=center><img width="650" height="400" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/6b33f6cf-8f56-4847-8b4d-59dd2247ebe6"/></div>

### Step 2-2.針對rbf核函數的參數C選擇
在rbf上可以得知由於rbf是一個可以讓模型較為複雜的核函數，因此隨著C的增加，再超過9之後test的表現便下降，代表此時模型開始出現overfitting的狀況，因此在下一步的Gamma參數選擇便挑選C=7，盡量同時減少underfitting和overfitting的情況。
<div align=center><img width="650" height="400" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/f9b0bd08-7873-4047-95ed-960665bba72d"/></div>

### Step 2-3.針對rbf核函數的參數Gamma選擇
在固定C=7的情況後針對不同的Gamma函數做挑選，可以觀察出在gamma=0.3時test有最佳的表現，而不論是更低還是更高時都可能會讓模型underfitting或overfitting，因此在後面的validation選擇便以gamma=0.3為固定參數。
<div align=center><img width="650" height="400" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/de314741-ff60-4321-ab70-facfb3f13957"/></div>

## Step 3.validation測試
為了進一步的測試參數以及模型的穩定度，因此這邊使用K-fold validation test來測試在不同train test的切片下，會不會大幅影響模型的表現

### Step 3-1.linear核函數validation測試
在linear核函數的模型上，可以觀察出大致的表現算穩定，在切片大時表現較差，到中間時表現最佳而隨著切片size更小表現逐漸下降，而在中間的部分是平坦的狀態且不會有很明顯的大起大落，代表這個模型算是一個穩定的模型，表現也是還不錯的平均正確率約98%
<div align=center><img width="650" height="400" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/0cde1591-c7c8-4e39-8300-3bed4c8db239"/></div>

### Step 3-2.rbf核函數validation測試
而在rbf核函數的模型上，雖然可以看出沒有linear模型那麼的平穩，然而正確率都維持在98~99%的區間內，因此仍然算是一個穩定的模型，同時即使只拿一半(也就是K=2)作為train來訓練模型，其在validation的平均準確率仍然有98%，代表這個模型也具有很好的效能
<div align=center><img width="650" height="400" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/48a2f13e-6453-4821-9d46-6471a4de7e8b"/></div>

## Step 4.最終參數以及模型選擇
在Step3進行過validation的測試後可以發現兩種不同核函數的模型都有足夠的穩定度，而在比較之後雖然linear的模型具有接近98%的正確率，然而rbf模型平均都可以擁有98%以上的準確度，因此最終選擇rbf模型作為SVM的核函數模型選擇，
此外，因為rbf模型在validation測試中以K=7具有較佳的表現，因此便以train/test的切片率大約為0.85/0.15作為最後測試模型的準確度，而最終根據下面的輸出可以得知獲得了98%的準確度，在254個資料中只有4個被分類錯誤，
也符合先前對模型的大致評估，即約為98%的正確率，後續在替換不同的random state時，正確率也是在98%附近以些微的數量變動，沒有過多影響，可以確認模型具有一定的穩定度。

<div align=center><img width="650" height="300" src="https://github.com/Zereo0317/Top_100_Company_ML/assets/142326772/89af8e8a-a87e-4721-938c-a201ac65673a"/></div>














