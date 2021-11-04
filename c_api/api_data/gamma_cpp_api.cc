/**
 * Copyright 2019 The Gamma Authors.
 *
 * This source code is licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 */
#include <fcntl.h>
#include <sys/stat.h>

#include <chrono>
#include <iostream>
#include <sstream>

#include "gamma_cpp_api.h"
#include "gamma_batch_result.h"
#include "gamma_config.h"
#include "gamma_doc.h"
#include "gamma_engine_status.h"
#include "gamma_response.h"
#include "gamma_table.h"
#include "search/gamma_engine.h"
#include "util/log.h"
#include "util/utils.h"

int CPPSearch(void *engine, tig_gamma::Request *request, tig_gamma::Response *response) {
  return static_cast<tig_gamma::GammaEngine *>(engine)->Search(*request, *response);
}

int CPPAddOrUpdateDoc(void *engine, tig_gamma::Doc *doc) {
    return static_cast<tig_gamma::GammaEngine *>(engine)->AddOrUpdate(*doc);        
}

int CPPAddOrUpdateDocs(void *engine, tig_gamma::Docs *docs, tig_gamma::BatchResult *results) {
    return static_cast<tig_gamma::GammaEngine *>(engine)->AddOrUpdateDocs(*docs, *results);
}

