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
(def response :group)
(def regressors
  (remove #{response} (ds/column-names data)))

;; ## Build pipelines
;; ### Input transforming pipelines
; In order for `:smile.classification` to work, categorical data needs to be transformed to numeric.
(def pipeline-fn
  (ml/pipeline
    (mm/categorical->number [response])
    (mm/set-inference-target response)))

(def pipeline-std-fn
  (ml/pipeline
    (mm/std-scale regressors {})
    (mm/categorical->number [response])
    (mm/set-inference-target response)))

;; ### Model building pipelines
(ml/hyperparameters
  :smile.classification/linear-discriminant-analysis)

; No hyperparameters.

(defn lda-piping-fn [pipeline]
  (ml/pipeline
    pipeline
    {:metamorph/id :model}
    (mm/model
      {:model-type :smile.classification/linear-discriminant-analysis})))

; ### Input_Transform->Model_Building pipelines
(def lda-pipe-fn
  (lda-piping-fn pipeline-fn))

(def lda-std-pipe-fn
  (lda-piping-fn pipeline-std-fn))

; #### View output of a fitted-pipeline
(-> data
    (ml/transform-pipe lda-std-pipe-fn
                       (ml/fit-pipe data lda-std-pipe-fn))
    :metamorph/data
    ds/shuffle
    ds/head)

;; ## Partition data
(def train-test
  (ds/split->seq data :bootstrap {:repeats 30}))

; Clojure's default `:bootstrapping` process takes a `:repeats` argument that is equivalent to `b`, number of bootstraps. Its training data proportion is determined by `:ratio`, whose default is `1`. The test data is the out-of-bag data, which would include the `1 - ratio` data when `:ratio` is not 1.

; ## Evaluate pipes
(def evaluate-pipes
  (ml/evaluate-pipelines
    [lda-pipe-fn lda-std-pipe-fn]
    train-test
    stats/cohens-kappa
    :accuracy
    {:other-metrices            [{:name :accuracy
                                  :metric-fn ml/classification-accuracy}
                                 {:name :mathews-cor-coef
                                  :metric-fn stats/mcc}]
     :return-best-pipeline-only false}))

; ## Extract models
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

; ### View model stats
(count models)
(-> models first :metric)
(-> models first :other-metrices
    (->> (map #(select-keys % [:name :metric]))))

(-> models second :metric)
(-> models second :other-metrices
    (->> (map #(select-keys % [:name :metric]))))

; Two models with exactly the same statistics. Meaning in this particular case, scaling and normalizing our data was not required for an improvement in the classification of groups.

(-> models first :fit-ctx second)                           ;look for :fit-ctx second has StdScaleTransform

; Notice in our first model's `:fit-ctx` we have a `:fit-std-xform`. That means this is our standardized pipeline. Might be interesting to keep this in mind for the next table.

(-> (model->ds (eval-maps models 2))
    (ds/rename-columns {:metric-1 :kappa                    ;TODO: extract from models
                        :metric-2 :accuracy
                        :metric-3 :mathews-cor-coef}))

; In Clojure, these metrics are rated on our `:test` data which is embedded in the partition data, `train-test`, and extracted in variable `models`.
;
; Everything's the same except compute time.

; ## Evaluations
; Above we can see our models' statistic on the test data. We might want to see how the best model fits on the full data.

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

; The `actual` variable looks like we are fitting a model, however, the code is running our data through the *input-transforming* pipeline as to get the appropriate mapping between group category and its respective numerical coding.

(ml/confusion-map->ds (ml/confusion-map predictions actual :none))

(-> models second :fit-ctx :model
    :target-categorical-maps :group :lookup-table)

(ml/classification-accuracy predictions actual)
(stats/cohens-kappa predictions actual)
(stats/mcc predictions actual)

; Woah! Something is wrong with the calculations. Let's see the datatypes:

(type (first predictions))
(type (first actual))

; One is a long type the other is a type double. These are not the same, which is why our kappa and mcc were so horribly low. Notice different datatypes' equivalencies and identities.

(= [1] `(1))
(identical? [1] `(1))

; Vectors `[]` are equivalent to lists ``()` (both in the sequence partition of Clojure.core), but not identical.

(= 1 1)
(= 1 1.0)
(identical? 1 1.0)

; But long 1 and double 1 are neither equivalent nor identical.
;
; I will map each more precise type (double) to the less granular type (long) as to ensure we are calculating the stats properly.

(ml/classification-accuracy (vec (map #(long %) predictions)) actual)
(stats/cohens-kappa (vec (map #(long %) predictions)) actual)
(stats/mcc (vec (map #(long %) predictions)) actual)

; It's better, however, it is interesting to see that adding an additional predictor, `:x2`, we aren't getting a bump in performance based on kappa and mcc.