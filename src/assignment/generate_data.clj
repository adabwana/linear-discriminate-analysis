(ns assignment.generate-data
  (:require
    [kixi.stats.distribution :as dist]
    [scicloj.ml.dataset :as ds]))

;; # Generate Data
(defn norm-dist [len]
  (take len (dist/normal {:mu -4 :sd 4})))

(defn make-normal-ds [len]
  {:x1    (take len (dist/normal {:mu 1 :sd 2 :location 3}))
   :x2    (norm-dist len)
   :group (take len (repeat "normal"))})

(defn gam-dist [len]
  (take len (dist/gamma {:shape 5 :rate 5})))

(defn make-gamma-ds [len]
  {:x1    (take len (dist/normal {:mu 0 :sd 2 :location 7}))
   :x2    (map #(* 8 %) (gam-dist len))
   :group (take len (repeat "gamma"))})

(defn log-norm-dist [len]
  (take len (dist/log-normal {:mu 1 :sd 15 :scale 0.5})))

(defn make-log-normal-ds [len]
  {:x1    (take len (dist/normal {:mu 0 :sd 2 :location -1}))
   :x2    (map #(* 2 %) (log-norm-dist len))
   :group (take len (repeat "log-normal"))})

(defn generate-data [len]
  (ds/dataset (merge-with concat (make-gamma-ds len) (make-normal-ds len) (make-log-normal-ds len))))

(def data
  (generate-data 200))