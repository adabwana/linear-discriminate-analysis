(ns assignment.r-interop
  (:require
    [assignment.generate-data :refer [data]]
    [assignment.lda :refer [lda-pipe-fn models]]
    [clojisr.v1.applications.plotting
     :refer [plot->svg]]
    [clojisr.v1.r :refer [r+]]
    [clojisr.v1.require :refer [require-r]]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.dataset :as ds]))

;; # R Interop & Plot

(require-r '[base :refer [summary]]
           '[ggplot2 :refer [ggplot aes geom_point
                             geom_contour theme_bw]])

(summary data)

(def dat (for [x (range -5 12 0.1)
               y (range -14 20 0.2)]
           {:x1 x :x2 y :group nil}))

(def contour-data
  (ds/dataset dat))

(def predictions
  (-> contour-data
      (ml/transform-pipe
        lda-pipe-fn
        (-> models second :fit-ctx))
      :metamorph/data
      :group
      vec))

(def lda-predict
  (ds/add-or-replace-column contour-data :group predictions))

; stat_contour(aes(x = X1, y = X2, z = y), data = lda_predict)
^kind/hiccup
(-> (ggplot :data data (aes :x 'x1 :y 'x2 :color 'group))
    (r+ (geom_point))
    (r+ (geom_contour :data lda-predict (aes :x 'x1 :y 'x2 :z 'group) :col "black"))
    (r+ (theme_bw))
    plot->svg)

(comment                                                    ;reverse process
  (def dat
    (for [minx (Math/floor (apply min (:x1 data)))
          maxx (Math/floor (apply max (:x1 data)))
          miny (Math/floor (apply min (:x2 data)))
          maxy (Math/floor (apply max (:x2 data)))
          x (range (int minx) (int maxx) 0.1)
          y (range (int miny) (int maxy) 0.2)]
      {:x1 x :x2 y :group nil}))

  (def lookup-table
    (-> models second :fit-ctx :model
        :target-categorical-maps :group :lookup-table))

  (def lookup-table-invert
    (clojure.set/map-invert lookup-table))

  (def lda-predict
    (-> lda-pred-pre-transform
        (ds/add-or-replace-column
          :group (map #(get lookup-table-invert %)
                      (map int (:group lda-predict)))))))