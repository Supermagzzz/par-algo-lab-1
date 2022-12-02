#include <iostream>
#include <chrono>
#include <functional>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

using namespace std;

pair<int, int> partition(vector<int>& a, int l, int r) {
    int pivot = a[(l + r) / 2];
    int midLeft = l;
    for (int i = l; i < r; i++) {
        if (a[i] < pivot) {
            swap(a[i], a[midLeft]);
            midLeft++;
        }
    }
    int midRight = midLeft;
    for (int i = midLeft; i < r; i++) {
        if (a[i] == pivot) {
            swap(a[i], a[midRight]);
            midRight++;
        }
    }
    return {midLeft, midRight};
}

void quickSort(vector<int>& a, int l, int r) {
    if (l + 1 >= r) {
        return;
    }
    auto [l1, r1] = partition(a, l, r);
    quickSort(a, l, l1);
    quickSort(a, r1, r);
}

void quickSort(vector<int>& a) {
    quickSort(a, 0, (int)a.size());
}

void quickSortParallel(vector<int>& a, int l, int r, int limit) {
    if (l + 1 >= r) {
        return;
    }
    if (r - l < limit) {
        quickSort(a, l, r);
        return;
    }
    auto [l1, r1] = partition(a, l, r);
#pragma omp task shared(a)
    {
        quickSortParallel(a, l, l1, limit);
    }
#pragma omp task shared(a)
    {
        quickSortParallel(a, r1, r, limit);
    }
}

void quickSortParallel(vector<int>& a, int limit) {
    omp_set_num_threads(4);
    omp_set_nested(2);
#pragma omp parallel shared(a)
    {
#pragma omp single
        {
            quickSortParallel(a, 0, (int)a.size(), limit);
        }
    }
}

vector<int> gen(int n) {
    vector<int> v(n);
    for (int &e : v) {
        e = (int)rand();
    }
    return v;
}

void benchmark() {
    double sum_simple = 0;
    double sum_parallel = 0;
    for (int i = 0; i < 5; i++) {
        int n = 1e8;
        auto arr = gen(n);
        auto sort_simple = arr;
        auto sort_parallel = arr;
        std::chrono::steady_clock::time_point begin_simple = std::chrono::steady_clock::now();
        quickSort(sort_simple);
        std::chrono::steady_clock::time_point end_simple = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point begin_par = std::chrono::steady_clock::now();
        quickSortParallel(sort_parallel, n / log(n));
        std::chrono::steady_clock::time_point end_par = std::chrono::steady_clock::now();

        auto simple_time = std::chrono::duration_cast<std::chrono::microseconds>(end_simple - begin_simple).count();
        auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end_par - begin_par).count();
        cout << simple_time / 1000.0 << " " << parallel_time / 1000.0 << endl;

        sum_simple += simple_time / 1e6;
        sum_parallel += parallel_time / 1e6;

        assert(sort_parallel == sort_simple);
    }
    cout << "avg" << endl;
    cout << sum_simple / sum_parallel << endl;
}

void stress() {
    while (true) {
        auto arr = gen(1e5);
        auto real_sort = arr;
        auto my_sort = arr;
        quickSortParallel(my_sort, 1e5 / 20);
        sort(real_sort.begin(), real_sort.end());
        if (my_sort != real_sort) {
            cout << "NO" << endl;
            exit(0);
        }
        cout << "YES" << endl;
    }
}

int main() {
    benchmark();
}
