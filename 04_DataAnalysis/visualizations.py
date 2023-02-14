#region import

#endregion

class visualizations:
    def confusion_matrix(y_pred, y_test, acc_balanced, absolute_or_relative_values, title, label_mapping = None, save_path):
        # Visualize Confusion Matrix with absolute values
        fig, ax = plt.subplots(figsize=(10, 5))
        mat = confusion_matrix(y_test, y_pred)
        if absolute_or_relative_values == "relative":
            mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]

        sns.heatmap(mat, annot=True, annot_kws={'size': 10}, cmap=plt.cm.Greens, fmt='d', cbar=False, linewidths=0.2)
        plt.title(title)
        plt.suptitle('Accuracy: {0:.3f}'.format(acc_balanced), fontsize=10)

        # try text
        params = {'axes.labelsize': 8,
                  'text.fontsize': 6,
                  'legend.fontsize': 7,
                  'xtick.labelsize': 6,
                  'ytick.labelsize': 6,
                  'text.usetex': True,  # <-- There
                  'figure.figsize': fig_size,
                  }
        rcParams.update(params)
        title(r"""\Huge{Big title !} \newline \tiny{Small subtitle !}""")

        # add xticks and yticks from label_mapping (which is a dictionary)
        tick_marks = np.arange(len(label_mapping)) + 0.5
        if label_mapping is not None:
            plt.xticks(tick_marks, label_mapping.values(), rotation=45)
            plt.yticks(tick_marks, label_mapping.values(), rotation=0)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(save_path)
        plt.show()

