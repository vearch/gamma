/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gamma_request.h"
#include "gamma_response.h"
#include "gamma_doc.h"
#include "gamma_docs.h"
#include "gamma_batch_result.h"

// Here are some corresponding C++ interfaces in c_api/gamma_api.h

int CPPSearch(void *engine, tig_gamma::Request *request, tig_gamma::Response *response);

int CPPAddOrUpdateDoc(void *engine, tig_gamma::Doc *doc);

int CPPAddOrUpdateDocs(void *engine, tig_gamma::Docs *docs, tig_gamma::BatchResult *results);

