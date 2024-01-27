(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []                                              ;for clay 63
  (clay/make!
    {:format              [:quarto :html]
     :book                {:title "Linear Discriminate Analysis"}
     :base-source-path    "src"
     :base-target-path    "docs"                            ;default
     :subdirs-to-sync     ["notebooks" "data"]
     :clean-up-target-dir true
     :source-path         ["index.clj"                      ;index.md
                           "assignment/generate_data.clj"
                           "assignment/visualize_data.clj"
                           "assignment/lda_univariate.clj"
                           "assignment/lda.clj"
                           "assignment/r_interop.clj"]}))

(defn build-book []                                         ;for clay >63
  (clay/make!
    {:base-source-path    "src"
     :base-target-path    "docs"                            ;default
     :title               "Linear Discriminate Analysis"
     ;:page-config ;configured in clay.edn
     :subdirs-to-sync     ["notebooks" "data"]
     :clean-up-target-dir true
     :quarto-book-config
     {:format               [:quarto :html]
      :chapter-source-paths ["index.clj"                    ;index.md
                             "assignment/generate_data.clj"
                             "assignment/visualize_data.clj"
                             "assignment/lda_univariate.clj"
                             "assignment/lda.clj"
                             "assignment/r_interop.clj"]}}))


(comment
  ;with index.md clay wont find in src and complains about docs/_book
  (build)
  (build-book))