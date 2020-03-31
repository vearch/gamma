/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

package main

/*
#cgo CFLAGS : -Ilib/include
#cgo LDFLAGS: -Llib/lib -lgamma

#include "search_api.h"
*/
import "C"

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/spf13/cast"
	"github.com/tiglabs/baudengine/util"
	"os"
	"strings"
	"time"
	"unsafe"
)

func main() {

	engine := initEngineAndTable()

	var err error
	var profile *os.File
	if profile, err = os.OpenFile("/root/wxd/feat_dir/sku_url_cid0_1.txt", os.O_RDONLY, os.ModePerm); err != nil {
		panic(err)
	}
	defer profile.Close()

	var feature *os.File
	if feature, err = os.OpenFile("/root/wxd/feat_dir/sku_url_cid0_1.txt", os.O_RDONLY, os.ModePerm); err != nil {
		panic(err)
	}
	defer feature.Close()

	profileScanner := bufio.NewScanner(profile)
	profileScanner.Split(bufio.ScanLines)

	featureScanner := bufio.NewScanner(profile)
	featureScanner.Split(bufio.ScanLines)

	i := 0
	for profileScanner.Scan() && featureScanner.Scan() {
		i++
		fmt.Println("index " + cast.ToString(i))
		insertDocument(engine, profileScanner.Text(), featureScanner.Text())
	}

	go func() {
		rc := C.BuildIndex(engine)
		if rc != 0 {
			fmt.Println(fmt.Sprintf("build index err response code:[%d]", rc))
		}
	}()
	for {
		s := C.GetStatus(engine)
		fmt.Println(fmt.Sprintf("index: status is %d", int(s)))

		if int(s) == 2 {
			fmt.Println("index ok")
			break
		}

		time.Sleep(3 * time.Second)
	}

	vqs := C.MakeVectorQuerys(C.int(1))

	featureScanner.Scan()

	C.SetVectorQuery(vqs, C.int(i), C.MakeVectorQuery(byteArrayStr("abc"), byteArray([]byte(featureScanner.Text())), C.double(*util.PFloat64(0)), C.double(util.PFloat64(100000000)), C.double(util.PFloat64(1))))

	req := C.MakeRequest(C.int(10),
		vqs, C.int(1),
		nil, C.int(0),
		nil, C.int(0),
		nil, C.int(0))

	rep := C.Search(engine, req)
	defer C.DestroyResponse(rep)

	fmt.Println(rep)

}

func insertDocument(engine unsafe.Pointer, profile, feature string) {
	split := strings.Split(profile, "\t")

	arr := C.MakeFields(C.int(5))
	C.SetField(arr, C.int(0), C.MakeField(byteArrayStr("sku"), valueToByte(cast.ToInt64(split[0])), LONG))
	C.SetField(arr, C.int(1), C.MakeField(byteArrayStr("_id"), byteArrayStr(split[1]), STRING))
	C.SetField(arr, C.int(2), C.MakeField(byteArrayStr("cid1"), valueToByte(cast.ToInt32(split[2])), INT))
	C.SetField(arr, C.int(3), C.MakeField(byteArrayStr("cid2"), valueToByte(cast.ToInt32(split[3])), INT))
	C.SetField(arr, byteArray([]byte(feature)), VECTOR)

	cDoc := &C.struct_Doc{fields: arr, fields_num: C.int(5)}
	defer C.DestroyDoc(cDoc)

	if resp := C.AddDoc(engine, (*C.struct_Doc)(unsafe.Pointer(cDoc))); resp != 0 {
		panic("index err .......")
	}
}

func initEngineAndTable() unsafe.Pointer {
	engine := C.Init(C.MakeConfig(byteArrayStr("files")))
	C.CreateTable(engine, makeTable())
	return engine
}

func makeTable() *C.struct_Table {
	fields_vec := []string{"sku", "_id", "cid1", "cid2"}
	fields_type := []interface{}{LONG, STRING, INT, INT}
	field_infos := C.MakeFieldInfos(C.int(len(fields_vec)))
	for i := 0; i < len(fields_vec); i++ {
		C.SetFieldInfo(field_infos, i, C.MakeFieldInfo(byteArrayStr(fields_vec[i]), fields_type[i]))
	}

	vectors_info := C.MakeVectorInfos(1)
	vectors_info.SetVectorInfo(vectors_info, 0, C.MakeVectorInfo(byteArrayStr("abc"), FLOAT, C.int(512), byteArrayStr("model")))

	table := &C.struct_Table{name: byteArrayStr("abc"), field_infos, C.int(len(fields_vec)), vectors_info, C.int(1)}

	return table
}

//util tool

var INT, LONG, FLOAT, DOUBLE, STRING, VECTOR C.enum_DataType = C.INT, C.LONG, C.FLOAT, C.DOUBLE, C.STRING, C.VECTOR

var empty = []byte{0}

//get cgo byte array
func byteArray(bytes []byte) *C.struct_ByteArray {
	if len(bytes) == 0 {
		return C.MakeByteArray((*C.char)(unsafe.Pointer(&empty[0])), C.int(len(bytes)))
	}

	return C.MakeByteArray((*C.char)(unsafe.Pointer(&bytes[0])), C.int(len(bytes)))
}

func byteArrayStr(str string) *C.struct_ByteArray {
	return byteArray([]byte(str))
}

func newField(name string, value []byte, typed C.enum_DataType) *C.struct_Field {
	return C.MakeField(byteArrayStr(name), byteArray(value), typed)
}

func newFieldBySource(name string, value []byte, source string, typed C.enum_DataType) *C.struct_Field {
	result := newField(name, value, typed)
	if source != "" {
		result.source = byteArrayStr(source)
	}
	return result
}

func valueToByte(fa interface{}) ([]byte, error) {
	buf := &bytes.Buffer{}
	buf.Reset()
	if err := binary.Write(buf, binary.LittleEndian, fa); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
