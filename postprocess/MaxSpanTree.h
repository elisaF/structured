#include <vector>
#include <iostream>

namespace trees {
    class MaxSpanTree {
    public:
        MaxSpanTree();

        ~MaxSpanTree();

        std::vector <std::vector<double> > get_tree(std::vector <std::vector<double> > str_scores);
    };
}
