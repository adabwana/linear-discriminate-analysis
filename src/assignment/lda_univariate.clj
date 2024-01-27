(ns assignment.lda-univariate
  (:require
    [assignment.generate-data :refer [data]]
    [fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [tech.v3.datatype.functional :as dfn]))

(def train-test
  (ds/split->seq (ds/drop-columns data :x2) :kfold {:ratio [0.8 0.2] :k 5 :seed 123}))

(def training (:train (first train-test)))
(def testing (:test (first train-test)))

;; Get group counts, estimate prior probabilities and centroids for each cluster
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

(defn discriminant-score [x mu var pi]
  (+ (- (* x (/ mu var)) (/ (Math/pow mu 2) (* 2 var))) (Math/log pi)))

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

(def pred-train
  (let [data (vec (:x1 training))]
    (vec (:predict (map-predict data)))))

(def actual-train
  (vec (:group training)))

(ml/confusion-map->ds (ml/confusion-map pred-train actual-train :none))

(def pred-test
  (let [data (vec (:x1 testing))]
    (vec (:predict (map-predict data)))))

(def actual-test
  (vec (:group testing)))

(ml/confusion-map->ds (ml/confusion-map pred-test actual-test :none))

(ml/classification-accuracy pred-test actual-test)
(stats/cohens-kappa pred-test actual-test)
(stats/mcc pred-test actual-test)
