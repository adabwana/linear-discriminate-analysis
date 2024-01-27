(ns assignment.lda
  (:require
    [assignment.generate-data :refer [data]]
    [calc-metric.patch]
    [fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]
    [utils.helpful-extracts :refer [eval-maps model->ds]]))

;; # Linear Discriminate Analysis
(comment                                                    ;if testing binary response
  (def data-subset
    (ds/select-rows data #(not= (:group %) "gamma"))))

(def response :group)
(def regressors
  (remove #{response} (ds/column-names data)))

;; ## Build pipelines
;; ### Generalized
(def pipeline-fn
  (ml/pipeline
    (mm/categorical->number [response])
    (mm/set-inference-target response)))

(def pipeline-std-fn
  (ml/pipeline
    (mm/std-scale regressors {})
    (mm/categorical->number [response])
    (mm/set-inference-target response)))

;; ### Specified
(ml/hyperparameters
  :smile.classification/linear-discriminant-analysis)
; No hyperparameters.

(defn lda-piping-fn [pipeline]
  (ml/pipeline
    pipeline
    {:metamorph/id :model}
    (mm/model
      {:model-type :smile.classification/linear-discriminant-analysis})))

(def lda-pipe-fn
  (lda-piping-fn pipeline-fn))
(def lda-std-pipe-fn
  (lda-piping-fn pipeline-std-fn))

(-> data
    (ml/transform-pipe lda-std-pipe-fn (ml/fit-pipe data lda-std-pipe-fn))
    :metamorph/data
    :group)

;; ## Partition data
(def train-test
  (ds/split->seq data :kfold {:ratio [0.8 0.2] :k 5}))

(def evaluate-pipes
  (ml/evaluate-pipelines
    [lda-pipe-fn lda-std-pipe-fn]
    train-test
    stats/cohens-kappa
    :accuracy
    {:other-metrices            [{:name :accuracy :metric-fn ml/classification-accuracy}
                                 {:name :mathews-cor-coef :metric-fn stats/mcc}]
     :return-best-pipeline-only false}))

(def models
  (->> evaluate-pipes
       flatten
       (map
         #(hash-map :summary (ml/thaw-model (get-in % [:fit-ctx :model]))
                    :fit-ctx (:fit-ctx %)
                    :timing-fit (:timing-fit %)
                    :metric ((comp :metric :test-transform) %)
                    :other-metrices ((comp :other-metrices :test-transform) %)
                    :params ((comp :options :model :fit-ctx) %)
                    :pipe-fn (:pipe-fn %)))
       (sort-by :metric)))

(count models)
(-> models first :metric)
(-> models first :other-metrices
    (->> (map #(select-keys % [:name :metric]))))

(-> models second :metric)
(-> models second :other-metrices
    (->> (map #(select-keys % [:name :metric]))))

(-> models first :fit-ctx second)                           ;look for :fit-ctx second has StdScaleTransform

(-> (model->ds (eval-maps models 2))
    (ds/rename-columns {:metric-1 :kappa                    ;TODO: extract from models
                        :metric-2 :accuracy
                        :metric-3 :mathews-cor-coef}))

(def predictions
  (-> data
      (ml/transform-pipe
        lda-pipe-fn
        (-> models first :fit-ctx))
      :metamorph/data
      :group
      vec))

(def actual
  (-> data
      (ml/fit-pipe lda-pipe-fn)
      :metamorph/data
      :group
      vec))

(ml/confusion-map->ds (ml/confusion-map predictions actual :none))

(comment
  (-> models second :fit-ctx :model :target-categorical-maps :group))
