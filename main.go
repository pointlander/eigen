// Copyright 2021 The Eigen Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"regexp"
	"strings"
)

func main() {
	data, err := ioutil.ReadFile("84-0.txt")
	if err != nil {
		panic(err)
	}
	split := regexp.MustCompile(`[\s]+`)
	words := split.Split(string(data), -1)
	count, unique := 0, make(map[string]int)
	for _, word := range words {
		normalized := strings.ToLower(strings.TrimSpace(word))
		if _, ok := unique[normalized]; !ok {
			unique[normalized] = count
			count++
		}
	}
	fmt.Println(len(unique))
}
