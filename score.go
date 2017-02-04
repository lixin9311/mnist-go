package main

import "./gomat"

type score struct {
	e float32
	c bool
}

type score_counter struct {
	R        []score
	capacity int
	p        int
	n        int
	sum_c    int
	sum_e    float64
}

func NewSC(cap int) *score_counter {
	sc := &score_counter{}
	sc.R = make([]score, cap)
	sc.capacity = cap
	return sc
}

func (sc *score_counter) Add(c bool, e float32) {
	if sc.p == sc.n {
		sc.n++
	} else {
		if sc.n != sc.capacity {
			panic("n cap mismatch!")
		}
		if sc.R[sc.p].c {
			sc.sum_c--
		}
		sc.sum_e -= float64(sc.R[sc.p].e)
	}
	sc.R[sc.p].c = c
	sc.R[sc.p].e = e
	if c {
		sc.sum_c++
	}

	sc.sum_e += float64(e)
	sc.p++
	if sc.p == sc.capacity {
		sc.p = 0
	}
}

func (sc *score_counter) Update_score(y, c, e *gomat.Matrix) {
	if y.Row() != c.Row() || y.Row() != e.Row() {
		panic("row mismatch!")
	}
	for i := 0; i < y.Row(); i++ {
		classify := y.Get(i, 0) == c.Get(i, 0)
		sc.Add(classify, e.Get(i, 0))
	}
}
