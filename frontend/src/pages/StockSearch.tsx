import React, { useState } from 'react';
import { Input, Card, List, Button } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Search } = Input;

const StockSearch: React.FC = () => {
  const [stocks, setStocks] = useState<any[]>([]);
  const navigate = useNavigate();

  const handleSearch = async (value: string) => {
    // 调用API搜索股票
    console.log('搜索:', value);
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <Card title="股票搜索">
        <Search
          placeholder="输入股票代码或名称"
          enterButton={<SearchOutlined />}
          size="large"
          onSearch={handleSearch}
        />
        
        <List
          style={{ marginTop: 24 }}
          dataSource={stocks}
          renderItem={(item: any) => (
            <List.Item
              actions={[
                <Button
                  type="link"
                  onClick={() => navigate(`/stock/${item.symbol}`)}
                >
                  查看详情
                </Button>
              ]}
            >
              <List.Item.Meta
                title={`${item.symbol} - ${item.name}`}
                description={item.industry}
              />
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default StockSearch;
