(ns assignment.lda
  (:require
    [assignment.generate-data :refer [data]]
    [calc-metric.patch]
    ;[fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]))

;; # Linear Discriminate Analysis
(def data-subset
  (ds/select-rows data #(not= (:group %) "gamma")))

(def response :group)
(def regressors
  (remove #{response} (ds/column-names data-subset)))

;; ## Build pipelines
;; ### Generalized
(def pipeline-vanilla-fn
  (ml/pipeline
    (mm/categorical->number [response])
    (mm/set-inference-target response)))

(-> (pipeline-vanilla-fn {:metamorph/data data-subset :metamorph/mode :fit})
    :metamorph/data)

;; ### Specified
(ml/hyperparameters :smile.classification/linear-discriminant-analysis)
; No hyperparameters.

(def lda-vanilla-pipe-fn
  (ml/pipeline
    pipeline-vanilla-fn
    {:metamorph/id :model}
    (mm/model {:model-type :smile.classification/linear-discriminant-analysis})))

(:metamorph/data (ml/fit-pipe data-subset lda-vanilla-pipe-fn))

;; ## Partition data
(def train-test
  (ds/split->seq data-subset :kfold {:ratio [0.8 0.2] :k 5}))

(comment                                                    ;will not run. :message "invalid type. everything above is using a subset of data with two categories. i did this because i thought maybe smile's lda did not want to work with more than binary responses
  (ml/evaluate-pipelines
    lda-vanilla-pipe-fn
    train-test
    ml/classification-accuracy
    :accuracy
    {:return-best-pipeline-only false}))
