// Copyright 2021 The Eigen Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/cmplx"
	"math/rand"
	"regexp"
	"sort"
	"strings"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

var (
	// Mode is the mode of the word vector algorithim
	Mode = flag.String("mode", "gradient", "the mode")
)

func normalize(a string) string {
	return strings.ToLower(strings.Trim(a, " \n\r\t.?!,;"))
}

func main() {
	flag.Parse()

	data, err := ioutil.ReadFile("84-0.txt")
	if err != nil {
		panic(err)
	}
	split := regexp.MustCompile(`[\s]+`)
	words := split.Split(string(data), -1)
	words = words[:4096]
	count, unique := 0, make(map[string]int)
	for _, word := range words {
		normalized := normalize(word)
		if _, ok := unique[normalized]; !ok {
			unique[normalized] = count
			count++
		}
	}
	size := len(unique)
	fmt.Println(size)

	var wordVectors [][]float64
	if *Mode == "gradient" {
		wordVectors = gradient(words, unique)
	} else if *Mode == "gonum" {
		wordVectors = gonum(words, unique)
	} else {
		flag.Usage()
		return
	}

	type Word struct {
		Key   string
		Value int
		Rank  float64
	}
	ranked := make([]Word, 0, 8)

	query := "good"
	good := wordVectors[unique[query]]
	max, maxKey, maxValue := 0.0, "", 0
	for key, value := range unique {
		sum, x, y := 0.0, 0.0, 0.0
		for i, a := range good {
			sum += wordVectors[value][i] * a
		}

		for _, a := range good {
			x += a * a
		}
		x = math.Sqrt(x)

		for _, a := range wordVectors[value] {
			y += a * a
		}
		y = math.Sqrt(y)

		sum /= x * y

		ranked = append(ranked, Word{
			Key:   key,
			Value: value,
			Rank:  sum,
		})
		if sum > max && key != query {
			max, maxKey, maxValue = sum, key, value
		}
	}
	fmt.Println(maxKey, maxValue)

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Rank < ranked[j].Rank
	})

	for _, word := range ranked {
		fmt.Println(word.Key, word.Rank)
	}
}

func gradient(words []string, unique map[string]int) [][]float64 {
	rand.Seed(1)

	size := len(unique)

	set := tf32.NewSet()
	set.Add("A", size, size)
	set.Add("X", size, size)
	set.Add("L", size, size)

	for i := range set.Weights[:2] {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 0)
			}
		} else {
			factor := float32(math.Sqrt(2 / float64(w.S[0])))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rand.NormFloat64())*factor)
			}
		}
	}

	set.Weights[2].X = set.Weights[2].X[:cap(set.Weights[2].X)]
	factor := float32(math.Sqrt(2 / float64(set.Weights[2].S[0])))
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if i == j {
				set.Weights[2].X[i*size+j] = float32(rand.NormFloat64()) * factor
			}
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	l1 := tf32.Mul(set.Get("A"), set.Get("X"))
	l2 := tf32.Mul(set.Get("L"), set.Get("X"))
	cost := tf32.Avg(tf32.Quadratic(l1, l2))

	alpha, eta, iterations := float32(.3), float32(.3), 1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for k, p := range set.Weights[:2] {
			for l, d := range p.D {
				deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
				p.X[l] += deltas[k][l]
			}
		}
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				if i == j {
					d := set.Weights[2].D[i*size+j]
					deltas[2][i*size+j] = alpha*deltas[2][i*size+j] - eta*d*scaling
					set.Weights[2].X[i*size+j] += deltas[2][i*size+j]
				}
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	wordVectors := make([][]float64, size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			wordVectors[i] = append(wordVectors[i], float64(set.Weights[1].X[j*size+i]))
		}
	}
	return wordVectors
}

func gonum(words []string, unique map[string]int) [][]float64 {
	size := len(unique)
	adjacency := sparse.NewDOK(size, size)
	for i := 1; i < len(words)-1; i++ {
		a := normalize(words[i-1])
		b := normalize(words[i])
		c := normalize(words[i+1])

		weight := adjacency.At(unique[a], unique[b])
		weight++
		adjacency.Set(unique[a], unique[b], weight)
		adjacency.Set(unique[b], unique[a], weight)

		weight = adjacency.At(unique[c], unique[b])
		weight++
		adjacency.Set(unique[c], unique[b], weight)
		adjacency.Set(unique[b], unique[c], weight)
	}
	fmt.Println("loaded adjacency matrix")

	var eig mat.Eigen
	ok := eig.Factorize(adjacency, mat.EigenRight)
	if !ok {
		panic("Eigendecomposition failed")
	}
	fmt.Println("computed eigenvectors")
	for i, value := range eig.Values(nil) {
		fmt.Println(i, cmplx.Abs(value))
	}
	vectors := mat.CDense{}
	eig.VectorsTo(&vectors)
	wordVectors := make([][]float64, size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			wordVectors[i] = append(wordVectors[i], cmplx.Abs(vectors.At(i, j)))
		}
	}
	return wordVectors
}
