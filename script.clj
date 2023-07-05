(ns script
  (:require
   [clojure.pprint :refer [pprint]]
   [clojure.edn :as edn]))

(def data (edn/read-string (slurp "./results.edn")))

(def sorted-data (sort-by (comp (juxt :vloss :last-erate) val) (:cols data)))


(pprint (take 11 sorted-data))

;; TSIs_13_25_13 == wma == t3 == trima