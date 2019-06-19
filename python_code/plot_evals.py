import deepdish as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PATH = "./automated_runs/pretrained/{}_layers/{}/models/evaluation/{}.h5"
INDEX = [[((2,"old",              "Wikiart_Elgammal_EQ_artist_train"), "Old Model"),
          ((0,"class_pretrained", "Wikiart_Elgammal_EQ_artist_train"), "Smallest Model"),
          ((1,"class_pretrained", "Wikiart_Elgammal_EQ_artist_train"), "Small Model"),
          ((2,"class_pretrained", "Wikiart_Elgammal_EQ_artist_train"), "Big Model"),
         ],
         [((2,"old",              "AwA2_vectors_train"), "Old Model"),
          ((0,"class_pretrained", "AwA2_vectors_train"), "Smallest Model"),
          ((1,"class_pretrained", "AwA2_vectors_train"), "Small Model"),
          ((2,"class_pretrained", "AwA2_vectors_train"), "Big Model"),
         ],
         [((2,"old",              "STL_label_train"), "Old Model"),
          ((0,"class_pretrained", "STL_label_train"), "Smallest Model"),
          ((1,"class_pretrained", "STL_label_train"), "Small Model"),
          ((2,"class_pretrained", "STL_label_train"), "Big Model"),
         ],
        ]

if __name__ == "__main__":

    for innerlist in INDEX:
        plt.clf()
        for i, (form, title) in enumerate(innerlist):
            d = dd.io.load(PATH.format(*form))
            filetitle = form[2]
            print "###" + filetitle + "###"
            if "clique_svm" in d:
                del d["clique_svm"]
            for ii, (heur, block) in enumerate(d.iteritems()): # ["svm", "clique_svm"]
                plt.subplot(len(innerlist), len(d), i*len(d)+ii+1)
                print "Subplot:", (len(innerlist), len(d), i*len(d)+ii+1)
                plt.title(title + "/" + heur)
                for name, df in block.iteritems(): # ["features", "projection"]
                    for legend, series in df.T.iteritems():
                        tmp, recall = zip(*list(series.iteritems()))
                        prec = [float(x[2:]) for x in zip(*tmp)[1]] # zip only indexable with py27
                        auc = np.trapz(prec, x=recall)
                        plt.plot(recall, prec, label="{}[{:.8f}]".format(name[:4], auc))#label="{}/{}[{:.1f}]".format(legend, name, auc))
                        

                plt.legend()
                #plt.tight_layout()
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(len(d)*8, len(INDEX)*8)
        plt.savefig("eval_for_{}.png".format(filetitle))
                