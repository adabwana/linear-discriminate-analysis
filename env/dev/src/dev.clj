(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
    {:format           [:quarto :html]
     :book             {:title "Linear Discriminate Analysis"}
     :base-source-path "src"
     ;:base-target-path "docs" ;default
     :subdirs-to-sync  ["notebooks" "data"]
     :source-path      ["index.clj"                          ;index.md
                        "assignment/generate_data.clj"
                        "assignment/visualize_data.clj"
                        "assignment/lda.clj"
                        "assignment/r-interop.clj"]}))

(comment
  ;with index.md clay wont find in src and complains about docs/_book
  (build))