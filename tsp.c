/*
 * Travelling Salesman Problem (TSP) Solver
 * Approach: Nearest Neighbor Heuristic + 2-opt Local Search
 * Author: SMILES 2026 Application
 *
 * Compile: gcc -O2 -o tsp tsp.c -lm
 * Run:     ./tsp
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_CITIES 20

/* ── Data structures ── */
typedef struct {
    double x, y;
    char name[32];
} City;

/* ── Distance helpers ── */
double euclidean(City a, City b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double tour_length(int *tour, int n, City *cities) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        int from = tour[i];
        int to   = tour[(i + 1) % n];
        total += euclidean(cities[from], cities[to]);
    }
    return total;
}

/* ── Nearest Neighbour heuristic ── */
void nearest_neighbour(int *tour, int n, City *cities, int start) {
    int visited[MAX_CITIES] = {0};
    tour[0]        = start;
    visited[start] = 1;

    for (int step = 1; step < n; step++) {
        int   current = tour[step - 1];
        int   nearest = -1;
        double best   = DBL_MAX;

        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                double d = euclidean(cities[current], cities[j]);
                if (d < best) { best = d; nearest = j; }
            }
        }
        tour[step]       = nearest;
        visited[nearest] = 1;
    }
}

/* ── 2-opt improvement ── */
void reverse_segment(int *tour, int i, int k) {
    while (i < k) {
        int tmp  = tour[i];
        tour[i]  = tour[k];
        tour[k]  = tmp;
        i++; k--;
    }
}

void two_opt(int *tour, int n, City *cities) {
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int k = i + 1; k < n; k++) {
                double d_before =
                    euclidean(cities[tour[i]],     cities[tour[i + 1]]) +
                    euclidean(cities[tour[k]],     cities[tour[(k + 1) % n]]);
                double d_after  =
                    euclidean(cities[tour[i]],     cities[tour[k]]) +
                    euclidean(cities[tour[i + 1]], cities[tour[(k + 1) % n]]);

                if (d_after < d_before - 1e-10) {
                    reverse_segment(tour, i + 1, k);
                    improved = 1;
                }
            }
        }
    }
}

/* ── Print helpers ── */
void print_tour(int *tour, int n, City *cities) {
    printf("\nOptimal Tour:\n");
    printf("  %-20s  %8s  %8s\n", "City", "X", "Y");
    printf("  %-20s  %8s  %8s\n", "----", "-", "-");
    for (int i = 0; i < n; i++) {
        City c = cities[tour[i]];
        printf("  %-20s  %8.2f  %8.2f\n", c.name, c.x, c.y);
    }
    /* close the loop */
    City c = cities[tour[0]];
    printf("  %-20s  %8.2f  %8.2f  (return)\n", c.name, c.x, c.y);
}

/* ── Main ── */
int main(void) {
    /* 10-city benchmark (approximate real-world coordinates scaled) */
    City cities[] = {
        { 0.00,  0.00, "City A"},
        { 3.00,  4.00, "City B"},
        { 6.00,  1.00, "City C"},
        { 9.00,  5.00, "City D"},
        { 2.00,  8.00, "City E"},
        { 5.00,  6.00, "City F"},
        { 8.00,  3.00, "City G"},
        { 1.00,  5.00, "City H"},
        { 7.00,  8.00, "City I"},
        { 4.00,  2.00, "City J"},
    };
    int n = sizeof(cities) / sizeof(cities[0]);

    printf("============================================\n");
    printf("  Travelling Salesman Problem Solver\n");
    printf("  Cities: %d\n", n);
    printf("============================================\n");

    int    best_tour[MAX_CITIES];
    int    tour[MAX_CITIES];
    double best_len = DBL_MAX;

    /* Try every city as starting point for NN, keep shortest */
    for (int start = 0; start < n; start++) {
        nearest_neighbour(tour, n, cities, start);
        two_opt(tour, n, cities);

        double len = tour_length(tour, n, cities);
        if (len < best_len) {
            best_len = len;
            for (int i = 0; i < n; i++) best_tour[i] = tour[i];
        }
    }

    print_tour(best_tour, n, cities);
    printf("\nTotal distance: %.4f units\n", best_len);
    printf("Algorithm: Nearest Neighbour + 2-opt\n");
    printf("============================================\n");

    return 0;
}
