(ns assignment.lda-univariate
  (:require
    [assignment.generate-data :refer [data]]
    [fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [tech.v3.datatype.functional :as dfn]))

; # Univariate LDA from Scratch
; ## Split data
; Use `:holdout` for a simple 1-split data.
(def train-test
  (ds/split->seq (ds/drop-columns data :x2)
                 :holdout {:ratio [0.8 0.2] :seed 0}))

(def training (:train (first train-test)))
(def testing (:test (first train-test)))

; ## Create dataset with key data
; Our key data includes mean, pooled-variance, and probabilities (called prior-prob) per group.

(def pooled-variance
  (-> training
      (ds/group-by [:group]
                   {:result-type :as-map})
      vals
      (->> (map :x1))
      stats/pooled-variance))

(def grouped-data
  (-> training
      (ds/group-by [:group])
      (ds/aggregate {:count #(count (% :x1))
                     :mean  #(dfn/mean (% :x1))})
      ;:variance #(stats/variance (% :x1))})
      (ds/add-column :pooled-variance pooled-variance)
      (ds/map-columns :prior-prob [:count] #(dfn// % (ds/row-count training)))
      (ds/select-columns #{:group :mean :pooled-variance :prior-prob})))

; ### View grouped data
grouped-data

; ## Create our key function
; In order to classify our data, we need to compare discriminate scores. In univariate linear discriminant analysis, the calculation is much simplified as compared to multivariate, which involves matrices.

(defn discriminant-score [x mu var pi]
  (+
    (- (* x (/ mu var)) (/ (Math/pow mu 2) (* 2 var)))
    (Math/log pi)))

; ## Implement labeling
; This is our classifying function.
(defn map-predict [dat]
  (-> (map
        (fn [data-point]
          (-> grouped-data
              (ds/map-columns
                :predict
                (ds/column-names grouped-data #{:mean :pooled-variance :prior-prob})
                (fn [mu var pi]
                  (discriminant-score data-point mu var pi)))
              (ds/order-by :predict :desc)
              (ds/select :group 0)
              :group
              vec))
        dat)
      ds/dataset
      (ds/rename-columns 0 {0 :predict})))

; ### How's map-predict work?
; Given a sequence of numbers, we are finding the highest discriminant score and labeling the data points that corresponding group. For example, let's sniff test our expectations, such as data near 0 is log-normal, data near 3 is normal, and data near 7 is gamma.

grouped-data

(map-predict [0 1 2 3 4 5 6 7])

; The results seem close to expectations.

; ## Predictions vs Actual
; ### Training
(def pred-train
  (let [data (vec (:x1 training))]
    (vec (:predict (map-predict data)))))

(def actual-train
  (vec (:group training)))

(ml/confusion-map->ds (ml/confusion-map pred-train actual-train :none))

; Confusion matrices made in Clojure's machine learning library (scicloj) according to the prescribed order (predicted-labels [true-]labels), have the TRUE classes *horizontally*. The columns represent the *prediction values*.

(ml/classification-accuracy pred-train actual-train)
(stats/cohens-kappa pred-train actual-train)
(stats/mcc pred-train actual-train)

; ### Test
(def pred-test
  (let [dat (vec (:x1 testing))]
    (vec (:predict (map-predict dat)))))

(def actual-test
  (vec (:group testing)))

(ml/confusion-map->ds (ml/confusion-map pred-test actual-test :none))

(ml/classification-accuracy pred-test actual-test)
(stats/cohens-kappa pred-test actual-test)
(stats/mcc pred-test actual-test)

; Performance on test is less than training. This is not unexpected, in fact it's quite common. Test data statistic *may* be better than the training data in cases where 1) the data is rather normal (no outliers), 2) the model is most appropriate to the data (e.g. a linear relation is modeled linearly) and/or 3) data size is small allowing for higher variation of these statistics.
;
; As for this model, having Cohen's Kappa and Mathews Correlation Coefficient greater than .6 is encouraging if only because, if you recall our :x1 distributions by group (final plot in our "Visualize Data" section), there is heavy overlap between our three groups.