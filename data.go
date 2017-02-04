package main

import "math/rand"

func choose_random_samples(M int, idxs []int) {
	for i := range idxs {
		idxs[i] = rand.Intn(M)
	}
}

func get_seq_samples(M int, begin int, idxs []int) {
	for i := range idxs {
		idxs[i] = (begin + i) % M
	}
}

func dump_as_ppm(a float32, m, n, magnify int, filename string) {
	panic("not implemented")
}
