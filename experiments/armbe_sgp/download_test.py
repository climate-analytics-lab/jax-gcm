import unittest
from unittest import mock

import download


class DownloadCliTest(unittest.TestCase):
    @mock.patch("download.pull_datastream")
    @mock.patch("download.list_files", return_value=["one.cdf", "two.cdf"])
    def test_list_only_does_not_download(self, list_files, pull_datastream):
        result = download.main([
            "--userid", "user",
            "--token", "token",
            "--datastreams", "sgptestC1.b1",
            "--start", "2018-01-01",
            "--end", "2018-01-02",
            "--list-only",
        ])

        self.assertEqual(result, 0)
        list_files.assert_called_once_with(
            "user", "token", "sgptestC1.b1", "2018-01-01", "2018-01-02"
        )
        pull_datastream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
