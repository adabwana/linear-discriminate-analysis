(ns assignment.visualize-data
  (:require
    [aerial.hanami.templates :as ht]
    [assignment.generate-data :refer [data]]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.ml.dataset :as ds]
    [scicloj.noj.v1.vis.hanami :as hanami]))

;; # Visualize Data
(-> data
    ds/shuffle
    (ds/head 7))

(def cols-of-interest [:x1 :x2])

; ## Group: normal
(def norm-dat
  (-> data
      (ds/select-rows #(= (:group %) "normal"))))

^kind/vega
(let [dat (ds/rows norm-dat :as-maps)
      column-names cols-of-interest]
  {:data   {:values dat}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x     {:field {:repeat "column"}
                               :bin   {:steps [1 3]} :type "quantitative"}
                       :y     {:aggregate "count"}
                       :color {:field :group}}}})

(ds/info norm-dat)

; ## Group: gamma
(def gamma-dat
  (ds/select-rows data #(= (:group %) "gamma")))

^kind/vega
(let [dat (ds/rows gamma-dat :as-maps)
      column-names cols-of-interest]
  {:data   {:values dat}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x     {:field {:repeat "column"}
                               :bin   {:steps [1 3]} :type "quantitative"}
                       :y     {:aggregate "count"}
                       :color {:field :group}}}})

(ds/info gamma-dat)

; ## Group: log-normal
(def log-normal-dat
  (ds/select-rows data #(= (:group %) "log-normal")))

^kind/vega
(let [dat (ds/rows log-normal-dat :as-maps)
      column-names cols-of-interest]
  {:data   {:values dat}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x     {:field {:repeat "column"}
                               :bin   {:steps [1 3]} :type "quantitative"}
                       :y     {:aggregate "count"}
                       :color {:field :group}}}})

(ds/info log-normal-dat)

;; ## Full data
(-> data
    (hanami/plot ht/point-chart
                 {:X "x1" :Y "x2" :COLOR "group"}))

^kind/vega
(let [dat (ds/rows data :as-maps)
      column-names cols-of-interest]
  {:data   {:values dat}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x     {:field {:repeat "column"}
                               :bin   {:steps [1 3]} :type "quantitative"}
                       :y     {:aggregate "count"}
                       :color {:field :group}}}})

(comment
  (defn dist-range [dist]
    (-> (apply max dist)
        (-
          (apply min dist)))))

(comment                                                    ; live works, but wont render
  (hanami/hconcat norm-dat {}
                  [(hanami/histogram norm-dat :x1 {:nbins 20})
                   (hanami/histogram norm-dat :x2 {:nbins 20})])

  (hanami/hconcat gamma-dat {}
                  [(hanami/histogram gamma-dat :x1 {:nbins 20})
                   (hanami/histogram gamma-dat :x2 {:nbins 20})])

  (hanami/hconcat log-normal-dat {}
                  [(hanami/histogram log-normal-dat :x1 {:nbins 20})
                   (hanami/histogram log-normal-dat :x2 {:nbins 20})])

  ^kind/vega
  (let [data (ds/rows data :as-maps)]
    {:data     {:values data}
     :mark     "bar"
     :encoding {:x     {:field :x1 :bin {:steps [2 3]} :type "quantitative"}
                :y     {:aggregate "count"}
                :color {:field :group}}}))
