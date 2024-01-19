(ns assignment.visualize-data
  (:require
    [aerial.hanami.templates :as ht]
    [assignment.generate-data :refer [data]]
    [scicloj.ml.dataset :as ds]
    [scicloj.noj.v1.vis.hanami :as hanami]))

;; # Visualize Data
(ds/head data)

(defn dist-range [dist]
  (-> (apply max dist)
      (-
        (apply min dist))))

(-> (ds/select-rows data #(= (:group %) "normal"))
    (hanami/histogram :x2 {:nbins 20}))

(ds/info (ds/select-rows data #(= (:group %) "normal")))

(-> (:x2 (ds/select-rows data #(= (:group %) "normal")))
    dist-range)

(-> (ds/select-rows data #(= (:group %) "gamma"))
    (hanami/histogram :x2 {:nbins 20}))

(ds/info (ds/select-rows data #(= (:group %) "gamma")))

(-> (:x2 (ds/select-rows data #(= (:group %) "gamma")))
    dist-range)

(-> (ds/select-rows data #(= (:group %) "log-normal"))
    (hanami/histogram :x2 {:nbins 20}))

(ds/info (ds/select-rows data #(= (:group %) "log-normal")))

(-> (:x2 (ds/select-rows data #(= (:group %) "log-normal")))
    dist-range)

;; Scatter plot
(-> data
    (hanami/plot ht/point-chart
                 {:X "x1" :Y "x2" :COLOR "group"}))
