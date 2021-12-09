// Copyright 2021 The Eigen Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/cmplx"
	"regexp"
	"sort"
	"strings"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

func main() {
	normalize := func(a string) string {
		return strings.ToLower(strings.Trim(a, " \n\r\t.?!,;"))
	}
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
		fmt.Println(word.Key)
	}
}
